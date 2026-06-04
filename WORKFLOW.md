# Experiment Workflow

This project uses GitHub as the transfer point between the local machine and the
cloud GPU server.

## Overview

The workflow is a repeated loop:

```text
Local edit -> GitHub push -> Server pull -> Server run -> GitHub push results -> Local pull -> Local analysis
```

Use the local machine for code editing and result analysis. Use the cloud server
only for GPU execution.

## 1. Edit Code Locally

Work in the local project directory:

```bash
cd "/Users/liuyanhan/Documents/New project 2"
```

Modify benchmark code, plotting code, README files, or experiment settings.

Before pushing, check the local status:

```bash
git status --short
```

Commit and push local changes:

```bash
git add .
git commit -m "Update experiment code"
git push origin main
```

## 2. Pull Code on the Server

On the cloud server:

```bash
cd /root/autodl-tmp/Project_1
git pull origin main
cd llama_decode_verify_bench
```

If the server cannot access GitHub directly, use the server-provided GitHub
proxy for pulling:

```bash
git pull https://ghfast.top/https://github.com/liuyanhan0627/Project_1.git main
```

The currently available server proxy is:

```text
https://ghfast.top
```

## 3. Run the Experiment on the Server

Run the benchmark and save logs:

```bash
cd /root/autodl-tmp/Project_1/llama_decode_verify_bench
mkdir -p logs
PYTHONUNBUFFERED=1 bash run_llama8b_a100.sh 2>&1 | tee logs/run_llama_a100.log
```

For the long-context experiment:

```bash
cd /root/autodl-tmp/Project_1/llama_decode_verify_bench
mkdir -p logs
PYTHONUNBUFFERED=1 bash run_llama8b_a100_longctx.sh 2>&1 | tee logs/run_llama_a100_longctx.log
```

Expected generated files:

```text
llama_decode_verify_bench/results/
llama_decode_verify_bench/figures/
llama_decode_verify_bench/figures_longctx/
llama_decode_verify_bench/logs/
```

## 4. Push Server Results Back to GitHub

After the run finishes, push results from the server:

```bash
cd /root/autodl-tmp/Project_1
git status --short
git add llama_decode_verify_bench/results \
        llama_decode_verify_bench/figures \
        llama_decode_verify_bench/figures_longctx \
        llama_decode_verify_bench/logs
git commit -m "Add benchmark results"
git push origin main
```

If pushing through the proxy fails, set the official GitHub remote:

```bash
git remote set-url origin https://github.com/liuyanhan0627/Project_1.git
git push origin main
```

Use a GitHub personal access token if Git asks for a password.

## 5. Pull Results Locally

Back on the local machine:

```bash
cd "/Users/liuyanhan/Documents/New project 2"
git pull origin main
```

Then inspect the returned files:

```bash
find llama_decode_verify_bench -maxdepth 3 -type f | grep -E "results|figures|logs"
```

## 6. Analyze Locally

Use the returned CSV, figures, and logs for analysis:

```text
llama_decode_verify_bench/results/
llama_decode_verify_bench/figures/
llama_decode_verify_bench/logs/
```

After analysis, decide the next code or experiment change and repeat from step 1.

## Notes

- Do not commit Hugging Face tokens, GitHub tokens, or model cache files.
- The server may need a GitHub proxy for pulling, but the local machine should
  use the official GitHub origin when VPN is available.
- Keep logs for failed runs too; failed logs are useful for debugging the next
  iteration.
