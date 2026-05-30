#!/usr/bin/env python3
"""Benchmark single-token decoding against multi-token verification.

This script isolates the server-side verification effect used by speculative
decoding systems such as SpecEdge. It builds a fixed prefix KV cache, then
compares:

1. One-step autoregressive decoding: one forward over shape [1, 1].
2. Batched verification: one forward over shape [1, n].
3. Sequential decoding: n forwards over shape [1, 1], updating KV each time.

Only the target model forward pass is timed. Tokenization, prefill, random token
construction, and KV cache cloning are intentionally outside the timed region.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoConfig, AutoModelForCausalLM


DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark causal-LM decoding vs batched candidate verification."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local checkpoint path, e.g. meta-llama/Llama-3.1-8B-Instruct.",
    )
    parser.add_argument(
        "--output",
        default="results/llama_decode_verify.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=sorted(DTYPES),
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. This benchmark is intended for CUDA/A100.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help='Optional Transformers device_map, e.g. "auto". Leave unset for single-GPU .to(device).',
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Optional Transformers attention backend.",
    )
    parser.add_argument(
        "--prefix-lens",
        nargs="+",
        type=int,
        default=[512, 2048, 4096],
        help="Prefix lengths used to build the KV cache.",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        help="Candidate token counts for verification.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for each measured case.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Measured iterations for each case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for synthetic token ids.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Transformers.",
    )
    parser.add_argument(
        "--low-token-id",
        type=int,
        default=10,
        help="Lower bound for random token ids, used to avoid common special ids.",
    )
    return parser.parse_args()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize_ms(times_ms: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "std_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p90_ms": percentile(times_ms, 0.90),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def cache_to_legacy(cache: Any) -> Any:
    """Return a tuple-of-tuples cache when possible.

    Transformers versions differ: some causal LM models return legacy tuple caches,
    newer versions may return DynamicCache. The legacy format is widely accepted
    by model forward and is easy to clone without mutating the base prefix cache.
    """
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return cache


def clone_cache(cache: Any) -> Any:
    """Clone a KV cache so each timed trial starts from the same prefix length."""
    cache = cache_to_legacy(cache)
    if cache is None:
        return None
    if isinstance(cache, tuple):
        cloned_layers = []
        for layer in cache:
            if isinstance(layer, tuple):
                cloned_layers.append(tuple(t.clone() for t in layer))
            else:
                cloned_layers.append(layer.clone())
        return tuple(cloned_layers)
    if isinstance(cache, list):
        return [clone_cache(item) for item in cache]
    return copy.deepcopy(cache)


def timed_cuda(
    prepare: Callable[[], Any],
    run: Callable[[Any], Any],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, float]:
    """Benchmark run(prepare()) with prepare excluded from the timed region."""
    if device.type != "cuda":
        raise RuntimeError("CUDA timing is required for this benchmark.")

    for _ in range(warmup):
        state = prepare()
        synchronize_if_needed(device)
        with torch.inference_mode():
            result = run(state)
        synchronize_if_needed(device)
        del result
        del state

    gc.collect()
    synchronize_if_needed(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []

    for _ in range(repeat):
        state = prepare()
        synchronize_if_needed(device)
        with torch.inference_mode():
            start_event.record()
            result = run(state)
            end_event.record()
        synchronize_if_needed(device)
        times_ms.append(start_event.elapsed_time(end_event))
        del result
        del state

    return summarize_ms(times_ms)


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> AutoModelForCausalLM:
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model: {args.model}")
    print(f"Model type: {getattr(config, 'model_type', 'unknown')}")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    if args.device_map is None:
        model.to(args.device)
    return model


def make_random_tokens(
    *,
    length: int,
    vocab_size: int,
    low_token_id: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    high = max(low_token_id + 1, vocab_size)
    return torch.randint(
        low=low_token_id,
        high=high,
        size=(1, length),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


def build_prefix_cache(
    model: AutoModelForCausalLM,
    prefix_ids: torch.Tensor,
    device: torch.device,
) -> Any:
    synchronize_if_needed(device)
    with torch.inference_mode():
        outputs = model(input_ids=prefix_ids, use_cache=True)
    synchronize_if_needed(device)
    return cache_to_legacy(outputs.past_key_values)


def main() -> None:
    args = parse_args()
    if args.dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    if any(n < 1 for n in args.ns):
        raise ValueError("--ns must contain positive integers.")
    if any(prefix_len < 1 for prefix_len in args.prefix_lens):
        raise ValueError("--prefix-lens must contain positive integers.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark is intended for CUDA. Run it on the A100 server.")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    dtype = DTYPES[args.dtype]
    model = load_model(args, dtype)
    model_device = next(model.parameters()).device
    if model_device.type != "cuda":
        raise RuntimeError(f"Model is on {model_device}, expected CUDA.")

    vocab_size = int(getattr(model.config, "vocab_size"))
    if args.low_token_id >= vocab_size:
        raise ValueError(
            f"--low-token-id ({args.low_token_id}) must be smaller than vocab_size ({vocab_size})."
        )
    max_prefix_len = max(args.prefix_lens)
    max_n = max(args.ns)
    generator = torch.Generator(device=model_device)
    generator.manual_seed(args.seed)
    prefix_pool = make_random_tokens(
        length=max_prefix_len,
        vocab_size=vocab_size,
        low_token_id=args.low_token_id,
        device=model_device,
        generator=generator,
    )
    candidate_pool = make_random_tokens(
        length=max_n,
        vocab_size=vocab_size,
        low_token_id=args.low_token_id,
        device=model_device,
        generator=generator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "device_name",
        "dtype",
        "attn_implementation",
        "prefix_len",
        "n",
        "warmup",
        "repeat",
        "t_decode_1_median_ms",
        "t_decode_1_mean_ms",
        "t_decode_1_p90_ms",
        "t_verify_n_median_ms",
        "t_verify_n_mean_ms",
        "t_verify_n_p90_ms",
        "t_decode_seq_n_median_ms",
        "t_decode_seq_n_mean_ms",
        "t_decode_seq_n_p90_ms",
        "verify_vs_one_decode",
        "verify_per_token_ms",
        "amortized_vs_one_decode",
        "speedup_over_seq",
        "created_unix_time",
    ]

    device_name = torch.cuda.get_device_name(model_device)
    created_unix_time = int(time.time())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prefix_len in args.prefix_lens:
            print(f"\nBuilding prefix cache for prefix_len={prefix_len}")
            prefix_ids = prefix_pool[:, :prefix_len].contiguous()
            prefix_cache = build_prefix_cache(model, prefix_ids, model_device)
            one_token = candidate_pool[:, :1].contiguous()

            def prepare_decode_1() -> Any:
                return clone_cache(prefix_cache)

            def run_decode_1(past: Any) -> Any:
                return model(input_ids=one_token, past_key_values=past, use_cache=True)

            print("Measuring one-token decoding baseline")
            decode_1_stats = timed_cuda(
                prepare_decode_1,
                run_decode_1,
                warmup=args.warmup,
                repeat=args.repeat,
                device=model_device,
            )
            t_decode_1 = decode_1_stats["median_ms"]

            for n in args.ns:
                verify_tokens = candidate_pool[:, :n].contiguous()

                def prepare_verify() -> Any:
                    return clone_cache(prefix_cache)

                def run_verify(past: Any) -> Any:
                    return model(input_ids=verify_tokens, past_key_values=past, use_cache=True)

                def prepare_seq() -> Any:
                    return clone_cache(prefix_cache)

                def run_seq(past: Any) -> Any:
                    current_past = past
                    for i in range(n):
                        step_token = verify_tokens[:, i : i + 1]
                        outputs = model(
                            input_ids=step_token,
                            past_key_values=current_past,
                            use_cache=True,
                        )
                        current_past = outputs.past_key_values
                    return current_past

                print(f"Measuring n={n}: verification [1,{n}] and sequential {n}x[1,1]")
                verify_stats = timed_cuda(
                    prepare_verify,
                    run_verify,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    device=model_device,
                )
                seq_stats = timed_cuda(
                    prepare_seq,
                    run_seq,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    device=model_device,
                )

                t_verify = verify_stats["median_ms"]
                t_seq = seq_stats["median_ms"]
                row = {
                    "model": args.model,
                    "device_name": device_name,
                    "dtype": args.dtype,
                    "attn_implementation": args.attn_implementation or "transformers_default",
                    "prefix_len": prefix_len,
                    "n": n,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "t_decode_1_median_ms": t_decode_1,
                    "t_decode_1_mean_ms": decode_1_stats["mean_ms"],
                    "t_decode_1_p90_ms": decode_1_stats["p90_ms"],
                    "t_verify_n_median_ms": t_verify,
                    "t_verify_n_mean_ms": verify_stats["mean_ms"],
                    "t_verify_n_p90_ms": verify_stats["p90_ms"],
                    "t_decode_seq_n_median_ms": t_seq,
                    "t_decode_seq_n_mean_ms": seq_stats["mean_ms"],
                    "t_decode_seq_n_p90_ms": seq_stats["p90_ms"],
                    "verify_vs_one_decode": t_verify / t_decode_1,
                    "verify_per_token_ms": t_verify / n,
                    "amortized_vs_one_decode": (t_verify / n) / t_decode_1,
                    "speedup_over_seq": t_seq / t_verify,
                    "created_unix_time": created_unix_time,
                }
                writer.writerow(row)
                f.flush()

            del prefix_cache
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
