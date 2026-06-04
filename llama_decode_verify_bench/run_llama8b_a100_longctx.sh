#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-./.matplotlib_cache}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT="${OUTPUT:-results/llama3_1_8b_a100_longctx.csv}"
TITLE_PREFIX="${TITLE_PREFIX:-Llama-3.1-8B A100 Long Context}"

mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}"

python bench_long_context_verify.py \
  --model "${MODEL}" \
  --dtype bf16 \
  --attn-implementation sdpa \
  --prefix-lens 4096 32768 65536 98304 120000 130000 \
  --ns 32 64 128 256 512 \
  --prefill-chunk-size 2048 \
  --warmup 5 \
  --repeat 20 \
  --output "${OUTPUT}"

python plot_results.py \
  --input "${OUTPUT}" \
  --output-dir figures_longctx \
  --title-prefix "${TITLE_PREFIX}"
