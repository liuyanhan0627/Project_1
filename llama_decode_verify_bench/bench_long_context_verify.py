#!/usr/bin/env python3
"""Long-context benchmark for single-token decoding vs batched verification.

This variant is designed for very long prefixes, such as 130k tokens. It avoids
cloning the full prefix KV cache for each timing trial. Instead, it builds the
prefix cache once with chunked prefill, times target-model forwards, then crops
the cache back to the prefix length outside the timed region.

The sequential decoding baseline is estimated as `n * T_decode_1`. This keeps
the experiment practical at 128k context while preserving the comparison to
ordinary token-by-token decoding.
"""

from __future__ import annotations

import argparse
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
        description="Benchmark batched verification at long context lengths."
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model id or local checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default="results/llama3_1_8b_a100_longctx.csv",
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
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Transformers attention backend.",
    )
    parser.add_argument(
        "--prefix-lens",
        nargs="+",
        type=int,
        default=[4096, 32768, 65536, 98304, 120000, 130000],
        help="Prefix lengths used to build the KV cache.",
    )
    parser.add_argument(
        "--ns",
        nargs="+",
        type=int,
        default=[32, 64, 128, 256, 512],
        help="Candidate token counts for batched verification.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations for each measured case.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Measured iterations for each case.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=2048,
        help="Number of prefix tokens per chunk during prefill.",
    )
    parser.add_argument(
        "--max-context-len",
        type=int,
        default=None,
        help="Maximum allowed prefix_len + n. Defaults to model config max_position_embeddings.",
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


def synchronize(device: torch.device) -> None:
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


def get_cache_seq_len(cache: Any) -> int | None:
    if cache is None:
        return None
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    if isinstance(cache, tuple) and cache:
        first_layer = cache[0]
        if isinstance(first_layer, tuple) and first_layer:
            return int(first_layer[0].shape[-2])
    return None


def crop_cache(cache: Any, max_length: int) -> Any:
    """Crop a cache back to max_length and return the current cache object.

    DynamicCache supports in-place crop(). Legacy tuple caches need rebuilt tuple
    views. Cropping is deliberately done outside the timed CUDA event region.
    """
    if cache is None:
        return None
    if hasattr(cache, "crop"):
        cache.crop(max_length)
        return cache
    if isinstance(cache, tuple):
        cropped_layers = []
        for layer in cache:
            if isinstance(layer, tuple):
                cropped_tensors = []
                for tensor in layer:
                    if torch.is_tensor(tensor) and tensor.ndim >= 3:
                        cropped_tensors.append(tensor[..., :max_length, :])
                    else:
                        cropped_tensors.append(tensor)
                cropped_layers.append(tuple(cropped_tensors))
            elif torch.is_tensor(layer) and layer.ndim >= 3:
                cropped_layers.append(layer[..., :max_length, :])
            else:
                cropped_layers.append(layer)
        return tuple(cropped_layers)
    if isinstance(cache, list):
        return [crop_cache(item, max_length) for item in cache]
    raise TypeError(f"Unsupported cache type for cropping: {type(cache)!r}")


def timed_cuda(
    run: Callable[[], Any],
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
    reset: Callable[[], None] | None = None,
) -> dict[str, float]:
    if device.type != "cuda":
        raise RuntimeError("CUDA timing is required for this benchmark.")

    for _ in range(warmup):
        if reset is not None:
            reset()
        synchronize(device)
        with torch.inference_mode():
            result = run()
        synchronize(device)
        del result
        if reset is not None:
            reset()

    gc.collect()
    synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times_ms: list[float] = []

    for _ in range(repeat):
        if reset is not None:
            reset()
        synchronize(device)
        with torch.inference_mode():
            start_event.record()
            result = run()
            end_event.record()
        synchronize(device)
        times_ms.append(start_event.elapsed_time(end_event))
        del result
        if reset is not None:
            reset()

    return summarize_ms(times_ms)


def load_model(args: argparse.Namespace, dtype: torch.dtype) -> AutoModelForCausalLM:
    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation is not None:
        load_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model: {args.model}")
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
    return torch.randint(
        low=low_token_id,
        high=vocab_size,
        size=(1, length),
        device=device,
        generator=generator,
        dtype=torch.long,
    )


def build_prefix_cache_chunked(
    model: AutoModelForCausalLM,
    prefix_ids: torch.Tensor,
    *,
    chunk_size: int,
    device: torch.device,
) -> tuple[Any, float]:
    if chunk_size < 1:
        raise ValueError("--prefill-chunk-size must be positive.")

    decoder = getattr(model, "model", model)
    past = None
    synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    with torch.inference_mode():
        for start in range(0, prefix_ids.shape[1], chunk_size):
            end = min(start + chunk_size, prefix_ids.shape[1])
            chunk = prefix_ids[:, start:end].contiguous()
            outputs = decoder(input_ids=chunk, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            del outputs

    end_event.record()
    synchronize(device)
    return past, start_event.elapsed_time(end_event) / 1000.0


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

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    max_context_len = args.max_context_len or int(
        getattr(config, "max_position_embeddings", 131072)
    )
    max_prefix_len = max(args.prefix_lens)
    max_n = max(args.ns)
    if max_prefix_len + max_n > max_context_len:
        raise ValueError(
            f"max(prefix_len) + max(n) = {max_prefix_len + max_n} exceeds "
            f"max_context_len = {max_context_len}."
        )

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
        "max_context_len",
        "prefill_chunk_size",
        "prefix_len",
        "n",
        "warmup",
        "repeat",
        "prefill_time_s",
        "cache_seq_len",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "decode_seq_mode",
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
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(model_device)
            prefix_ids = prefix_pool[:, :prefix_len].contiguous()
            prefix_cache, prefill_time_s = build_prefix_cache_chunked(
                model,
                prefix_ids,
                chunk_size=args.prefill_chunk_size,
                device=model_device,
            )
            cache_seq_len = get_cache_seq_len(prefix_cache)
            print(
                f"Built prefix cache: seq_len={cache_seq_len}, "
                f"prefill_time={prefill_time_s:.2f}s"
            )

            cache_holder = {"cache": prefix_cache}

            def reset_prefix_cache() -> None:
                cache_holder["cache"] = crop_cache(cache_holder["cache"], prefix_len)
                current_len = get_cache_seq_len(cache_holder["cache"])
                if current_len != prefix_len:
                    raise RuntimeError(
                        f"Failed to reset prefix cache to {prefix_len}; got {current_len}."
                    )

            one_token = candidate_pool[:, :1].contiguous()

            def run_decode_1() -> Any:
                return model(
                    input_ids=one_token,
                    past_key_values=cache_holder["cache"],
                    use_cache=True,
                )

            print("Measuring one-token decoding baseline")
            decode_1_stats = timed_cuda(
                run_decode_1,
                warmup=args.warmup,
                repeat=args.repeat,
                device=model_device,
                reset=reset_prefix_cache,
            )

            t_decode_1 = decode_1_stats["median_ms"]

            for n in args.ns:
                verify_tokens = candidate_pool[:, :n].contiguous()

                def run_verify() -> Any:
                    return model(
                        input_ids=verify_tokens,
                        past_key_values=cache_holder["cache"],
                        use_cache=True,
                    )

                print(f"Measuring n={n}: verification [1,{n}], sequential baseline estimated")
                verify_stats = timed_cuda(
                    run_verify,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    device=model_device,
                    reset=reset_prefix_cache,
                )

                t_verify = verify_stats["median_ms"]
                t_seq_est = t_decode_1 * n
                row = {
                    "model": args.model,
                    "device_name": device_name,
                    "dtype": args.dtype,
                    "attn_implementation": args.attn_implementation or "transformers_default",
                    "max_context_len": max_context_len,
                    "prefill_chunk_size": args.prefill_chunk_size,
                    "prefix_len": prefix_len,
                    "n": n,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "prefill_time_s": prefill_time_s,
                    "cache_seq_len": cache_seq_len,
                    "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(model_device)
                    / (1024**3),
                    "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(model_device)
                    / (1024**3),
                    "decode_seq_mode": "estimated_n_times_decode_1",
                    "t_decode_1_median_ms": t_decode_1,
                    "t_decode_1_mean_ms": decode_1_stats["mean_ms"],
                    "t_decode_1_p90_ms": decode_1_stats["p90_ms"],
                    "t_verify_n_median_ms": t_verify,
                    "t_verify_n_mean_ms": verify_stats["mean_ms"],
                    "t_verify_n_p90_ms": verify_stats["p90_ms"],
                    "t_decode_seq_n_median_ms": t_seq_est,
                    "t_decode_seq_n_mean_ms": decode_1_stats["mean_ms"] * n,
                    "t_decode_seq_n_p90_ms": decode_1_stats["p90_ms"] * n,
                    "verify_vs_one_decode": t_verify / t_decode_1,
                    "verify_per_token_ms": t_verify / n,
                    "amortized_vs_one_decode": (t_verify / n) / t_decode_1,
                    "speedup_over_seq": t_seq_est / t_verify,
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
