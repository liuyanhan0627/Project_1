#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-./.matplotlib_cache}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
OUTPUT="${OUTPUT:-results/llama3_1_8b_a100.csv}"
TITLE_PREFIX="${TITLE_PREFIX:-Llama-3.1-8B A100}"

mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}"

python bench_decode_verify.py \
  --model "${MODEL}" \
  --dtype bf16 \
  --attn-implementation sdpa \
  --prefix-lens 512 2048 4096 \
  --ns 1 2 4 8 16 32 64 128 256 \
  --warmup 20 \
  --repeat 100 \
  --output "${OUTPUT}"

python plot_results.py \
  --input "${OUTPUT}" \
  --output-dir figures \
  --title-prefix "${TITLE_PREFIX}"
