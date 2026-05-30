#!/usr/bin/env python3
"""Plot benchmark results from bench_decode_verify.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decode/verify benchmark CSV.")
    parser.add_argument("--input", required=True, help="CSV produced by bench_decode_verify.py.")
    parser.add_argument("--output-dir", default="figures", help="Directory for PNG figures.")
    parser.add_argument("--title-prefix", default="", help="Optional title prefix.")
    return parser.parse_args()


def plot_lines(
    df: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for prefix_len, group in sorted(df.groupby("prefix_len")):
        group = group.sort_values("n")
        ax.plot(group["n"], group[y], marker="o", label=f"prefix_len={prefix_len}")
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("n candidate tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    required = {
        "prefix_len",
        "n",
        "verify_vs_one_decode",
        "verify_per_token_ms",
        "amortized_vs_one_decode",
        "speedup_over_seq",
        "t_decode_1_median_ms",
        "t_verify_n_median_ms",
        "t_decode_seq_n_median_ms",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.title_prefix} " if args.title_prefix else ""

    plot_lines(
        df,
        y="verify_vs_one_decode",
        ylabel="T_verify_n / T_decode_1",
        title=f"{prefix}Verification Time Relative to One Decode",
        output_path=out_dir / "verify_vs_one_decode.png",
    )
    plot_lines(
        df,
        y="verify_per_token_ms",
        ylabel="T_verify_n / n (ms/token)",
        title=f"{prefix}Amortized Verification Latency",
        output_path=out_dir / "verify_per_token_ms.png",
    )
    plot_lines(
        df,
        y="amortized_vs_one_decode",
        ylabel="(T_verify_n / n) / T_decode_1",
        title=f"{prefix}Amortized Verification Relative to One Decode",
        output_path=out_dir / "amortized_vs_one_decode.png",
    )
    plot_lines(
        df,
        y="speedup_over_seq",
        ylabel="T_decode_seq_n / T_verify_n",
        title=f"{prefix}Speedup over Sequential Decoding",
        output_path=out_dir / "speedup_over_seq.png",
    )
    plot_lines(
        df,
        y="t_verify_n_median_ms",
        ylabel="T_verify_n (ms)",
        title=f"{prefix}Total Verification Latency",
        output_path=out_dir / "verify_total_latency.png",
    )

    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
