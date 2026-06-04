# Llama Decode vs Verification Benchmark

This experiment validates one mechanism behind speculative decoding systems such
as SpecEdge: with the same prefix KV cache, verifying `n` candidate tokens in one
target-model forward pass can amortize model-weight access compared with
ordinary one-token decoding.

The default model is `meta-llama/Llama-3.1-8B-Instruct`, and the default run script is
set up for a single A100 GPU.

The benchmark compares three cases:

- `T_decode_1`: one autoregressive decoding step, input shape `[1, 1]`.
- `T_verify_n`: one verification forward, input shape `[1, n]`.
- `T_decode_seq_n`: `n` sequential decoding steps, each input shape `[1, 1]`.

Only model forward time is measured. Prefill, tokenizer work, random token
generation, CPU/GPU transfer, sampling, and KV cache cloning are outside the
timed region.

## Files

- `bench_decode_verify.py`: main benchmark script.
- `bench_long_context_verify.py`: long-context benchmark script for prefixes up to 130k tokens.
- `plot_results.py`: plots CSV results.
- `run_llama8b_a100.sh`: recommended Llama-3.1-8B run command for A100.
- `run_llama8b_a100_longctx.sh`: recommended long-context run command for A100.
- `requirements.txt`: Python dependencies.

## Setup

Install dependencies on the cloud server:

```bash
pip install -r requirements.txt
```

For the official Meta Llama checkpoint, log in with the Hugging Face account
that has already been granted model access:

```bash
hf auth login
hf auth whoami
```

## Run

### Short/Medium Context

From this directory:

```bash
bash run_llama8b_a100.sh
```

You can override the model or output path without editing the script:

```bash
MODEL=meta-llama/Llama-3-8B-Instruct OUTPUT=results/llama3_8b_a100.csv bash run_llama8b_a100.sh
```

For a local checkpoint:

```bash
MODEL=/path/to/local/llama-checkpoint OUTPUT=results/local_llama.csv bash run_llama8b_a100.sh
```

Or call the benchmark directly:

```bash
python bench_decode_verify.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bf16 \
  --attn-implementation sdpa \
  --prefix-lens 512 2048 4096 \
  --ns 1 2 4 8 16 32 64 128 256 \
  --warmup 20 \
  --repeat 100 \
  --output results/llama3_1_8b_a100.csv
```

The script defaults to `HF_ENDPOINT=https://hf-mirror.com` in `run_llama8b_a100.sh`.
If the mirror does not see your newly granted permission immediately, try the
official endpoint by running `HF_ENDPOINT=https://huggingface.co bash run_llama8b_a100.sh`.

### Long Context

For the 130k-token context experiment, run:

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 bash run_llama8b_a100_longctx.sh 2>&1 | tee logs/run_llama_a100_longctx.log
```

Default long-context settings:

```text
prefix_len = 4096, 32768, 65536, 98304, 120000, 130000
n = 32, 64, 128, 256, 512
prefill_chunk_size = 2048
warmup = 5
repeat = 20
```

The long-context script builds the prefix KV cache with chunked prefill and does
not clone the full 130k-token KV cache for every timing trial. Its sequential
decoding baseline is estimated as `n * T_decode_1`, which is recorded in the CSV
as `decode_seq_mode = estimated_n_times_decode_1`.

## Output Metrics

The CSV includes:

- `verify_vs_one_decode = T_verify_n / T_decode_1`
- `verify_per_token_ms = T_verify_n / n`
- `amortized_vs_one_decode = (T_verify_n / n) / T_decode_1`
- `speedup_over_seq = T_decode_seq_n / T_verify_n`

The long-context CSV also includes:

- `prefill_time_s`
- `cache_seq_len`
- `peak_memory_allocated_gb`
- `peak_memory_reserved_gb`
- `decode_seq_mode`

The key expected pattern is that `T_verify_n` grows with `n`, but much more
slowly than `T_decode_seq_n`; therefore `T_verify_n / n` should be lower than
the one-token decoding latency for sufficiently large `n`.
