#!/usr/bin/env python3
"""
plot_deltas.py — plot consecutive-step deltas for train vs submission.

Usage:
  python plot_deltas.py \
    --train-csv data/train.csv \
    --sub-csv submissions/submission_tft_lags_2.csv \
    --out-dir reports/plots \
    --time-col id \
    --abs \
    --from-date 2024-01-01
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_METRICS = ["valeur_CO", "valeur_NO2", "valeur_O3", "valeur_PM10", "valeur_PM25"]

def parse_args():
    ap = argparse.ArgumentParser(description="Plot consecutive-step deltas for train vs submission.")
    ap.add_argument("--train-csv", required=True, help="Path to training CSV.")
    ap.add_argument("--sub-csv",   required=True, help="Path to submission CSV.")
    ap.add_argument("--out-dir",   required=True, help="Directory to write PNG plots.")
    ap.add_argument("--time-col",  default="id",  help="Timestamp column name (default: id).")
    ap.add_argument("--metrics",   nargs="*", default=DEFAULT_METRICS,
                    help="Pollutant columns to analyze.")
    ap.add_argument("--dpi", type=int, default=140, help="Image DPI (default: 140).")
    ap.add_argument("--show", action="store_true", help="Show plots interactively as well.")
    ap.add_argument("--abs", dest="abs_delta", action="store_true",
                    help="Plot absolute deltas |Δ| instead of signed Δ.")
    ap.add_argument("--from-date", default="2024-01-01",
                    help="Keep rows with time >= this date (YYYY-MM-DD). Default: 2024-01-01")
    return ap.parse_args()

def coerce_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Available: {list(df.columns)}")
    out = df.copy()
    # Robust to 'YYYY-MM-DD HH' and full timestamps
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=False)
    if out[time_col].isna().any():
        bad = out.loc[out[time_col].isna(), time_col].index[:5]
        raise ValueError(f"Failed to parse some timestamps in '{time_col}'. "
                         f"First bad indices: {list(bad)}")
    return out

def filter_from_date(df: pd.DataFrame, time_col: str, from_date: str) -> pd.DataFrame:
    if from_date:
        cutoff = pd.to_datetime(from_date)
        df = df[df[time_col] >= cutoff].copy()
    return df

def find_present_metrics(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]

def consecutive_deltas(df: pd.DataFrame, time_col: str, metric: str, abs_delta: bool) -> pd.DataFrame:
    """Return [time_col, delta] for one metric (sorted, diffed)."""
    sub = df[[time_col, metric]].copy()
    sub = sub.sort_values(time_col).reset_index(drop=True)

    vals = pd.to_numeric(sub[metric], errors="coerce")
    d = vals.diff()

    if abs_delta:
        # absolute + guard against tiny negative epsilons
        d = d.abs().clip(lower=0)

    sub["delta"] = d
    sub = sub.dropna(subset=["delta"])
    return sub[[time_col, "delta"]]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load & parse time
    train = pd.read_csv(args.train_csv)
    sub   = pd.read_csv(args.sub_csv)

    train = coerce_time(train, args.time_col)
    sub   = coerce_time(sub,   args.time_col)

    # Keep only 2024+ (or user-specified)
    train = filter_from_date(train, args.time_col, args.from_date)
    sub   = filter_from_date(sub,   args.time_col, args.from_date)

    # Figure out which metrics exist in each, and intersect
    train_metrics = find_present_metrics(train, args.metrics)
    sub_metrics   = find_present_metrics(sub,   args.metrics)
    metrics = [m for m in train_metrics if m in sub_metrics]
    if not metrics:
        raise ValueError(f"No overlapping metrics to plot.\n"
                         f"Train has: {train_metrics}\nSubmission has: {sub_metrics}")

    # Matplotlib style tweaks for readability
    plt.rcParams.update({
        "figure.figsize": (12, 4),
        "axes.grid": True,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "legend.loc": "upper right",
        "lines.linewidth": 1.2,
    })

    label_delta = "|Δ| (abs change)" if args.abs_delta else "Δ value (current - previous)"
    title_suffix = " (|Δ|)" if args.abs_delta else " (Δ)"

    for metric in metrics:
        # Build deltas
        d_train = consecutive_deltas(train, args.time_col, metric, args.abs_delta)
        d_sub   = consecutive_deltas(sub,   args.time_col, metric, args.abs_delta)

        fig, ax = plt.subplots()
        ax.plot(d_train[args.time_col], d_train["delta"], label="Train Δ", alpha=0.9)
        ax.plot(d_sub[args.time_col],   d_sub["delta"],   label="Submission Δ", alpha=0.9)

        # Optional: band of ±3σ from train deltas (helps spot unrealistic moves)
        if len(d_train) > 5:
            std_tr = float(d_train["delta"].std(ddof=1))
            mean_tr = float(d_train["delta"].mean())
            ax.axhline(mean_tr, linestyle="--", alpha=0.6)
            ax.axhspan(mean_tr - 3*std_tr, mean_tr + 3*std_tr, alpha=0.1)

        ax.set_title(f"Consecutive-step change for {metric}{title_suffix}")
        ax.set_xlabel("Time")
        ax.set_ylabel(label_delta)
        ax.legend(ncol=2)

        # Nice datetime formatting
        fig.autofmt_xdate()

        out_path = os.path.join(args.out_dir, f"deltas_{metric}.png")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        if args.show:
            plt.show()
        plt.close(fig)

        print(f"Saved: {out_path}")

    print("\nDone. One PNG per metric written to:", args.out_dir)

if __name__ == "__main__":
    main()
