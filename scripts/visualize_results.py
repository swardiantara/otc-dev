"""
Generate comparison plots from test-set metrics.

Pipeline mirrors analyze_results.py:
  load → extract lr/seed → mean/std per (dataset,loss,lr) → best lr per
  (dataset,loss) → rank → plot.

Produces (saved under results/figures/):
  - bar_<metric>.pdf      : grouped bar chart (mean ± std error bars)
  - heatmap_<metric>.pdf  : loss × dataset heatmap (mean ± std annotations)
  - average_rank.pdf      : average rank across datasets and metrics

Usage:
    python -m scripts.visualize_results
    python -m scripts.visualize_results --metrics mae mse --datasets sst5 yelp
    python -m scripts.visualize_results --sort_metric mae --lower_is_better
    python -m scripts.visualize_results --output_dir results/my_figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.analyze_results import (
    load_metrics,
    extract_run_info,
    compute_mean_std,
    select_best_lr,
    compute_ranks,
    LOSS_ORDER,
    DEFAULT_METRICS_DIR,
    ANALYSIS_METRICS,
    LOWER_IS_BETTER,
)

ROOT_PATH = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = ROOT_PATH / "results" / "figures"

DATASET_ORDER = ["amazon_reviews", "snli", "sst5", "yelp"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_score": "Weighted F1",
    "mae": "MAE",
    "mse": "MSE",
    "kendalltau": "Kendall's τ",
    "off-by-1-accuracy": "OB1",
    "off-by-2-accuracy": "OB2",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _present_losses(index):
    present = [l for l in LOSS_ORDER if l in index]
    extra = [l for l in index if l not in LOSS_ORDER]
    return present + extra


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def grouped_bar(best: pd.DataFrame, metric: str, datasets: list, output_dir: Path) -> None:
    """Grouped bar chart using best-lr mean scores with std error bars."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in best.columns:
        print(f"  Skipping bar for '{metric}' — mean column not found.")
        return

    sub = best[best["dataset"].isin(datasets)][["dataset", "loss", mean_col, std_col]].copy()
    pivot_mean = sub.pivot(index="loss", columns="dataset", values=mean_col)
    pivot_std = sub.pivot(index="loss", columns="dataset", values=std_col)

    present_losses = _present_losses(pivot_mean.index)
    present_datasets = [d for d in DATASET_ORDER if d in pivot_mean.columns]
    pivot_mean = pivot_mean.loc[present_losses, present_datasets]
    pivot_std = pivot_std.loc[present_losses, present_datasets]

    n_losses = len(present_losses)
    n_datasets = len(present_datasets)
    x = np.arange(n_losses)
    width = 0.8 / n_datasets
    palette = sns.color_palette("tab10", n_datasets)

    fig, ax = plt.subplots(figsize=(max(10, n_losses * 1.2), 5))
    for i, dataset in enumerate(present_datasets):
        offset = (i - n_datasets / 2 + 0.5) * width
        vals = pivot_mean[dataset].values
        errs = pivot_std[dataset].fillna(0).values
        ax.bar(x + offset, vals, width * 0.9, yerr=errs, capsize=3,
               label=dataset, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(present_losses, fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Loss Function and Dataset")
    ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()

    out_path = output_dir / f"bar_{metric}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


def heatmap(best: pd.DataFrame, metric: str, datasets: list, output_dir: Path) -> None:
    """Loss × Dataset heatmap with mean ± std cell annotations."""
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in best.columns:
        print(f"  Skipping heatmap for '{metric}' — mean column not found.")
        return

    sub = best[best["dataset"].isin(datasets)][["dataset", "loss", mean_col, std_col]].copy()
    pivot_mean = sub.pivot(index="loss", columns="dataset", values=mean_col)
    pivot_std = sub.pivot(index="loss", columns="dataset", values=std_col)

    present_losses = _present_losses(pivot_mean.index)
    present_datasets = [d for d in DATASET_ORDER if d in pivot_mean.columns]
    pivot_mean = pivot_mean.loc[present_losses, present_datasets].astype(float)
    pivot_std = pivot_std.loc[present_losses, present_datasets].astype(float)

    # Build annotation strings: "mean\n±std"
    annot = np.empty(pivot_mean.shape, dtype=object)
    for i, loss in enumerate(pivot_mean.index):
        for j, ds in enumerate(pivot_mean.columns):
            m = pivot_mean.loc[loss, ds]
            s = pivot_std.loc[loss, ds]
            if pd.isna(m):
                annot[i, j] = "N/A"
            elif pd.isna(s) or s == 0:
                annot[i, j] = f"{m:.3f}"
            else:
                annot[i, j] = f"{m:.3f}\n±{s:.3f}"

    fig, ax = plt.subplots(
        figsize=(max(6, len(present_datasets) * 2.2), max(4, len(present_losses) * 0.85))
    )
    cmap = "RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn"
    sns.heatmap(
        pivot_mean, ax=ax, cmap=cmap, annot=annot, fmt="",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
        annot_kws={"size": 8},
    )
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — Loss × Dataset (mean ± std)")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Loss Function")
    fig.tight_layout()

    out_path = output_dir / f"heatmap_{metric}.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


def rank_plot(best: pd.DataFrame, metrics: list, datasets: list, output_dir: Path) -> None:
    """Average rank of each loss across datasets and metrics (lower = better overall)."""
    ranks = compute_ranks(best, metrics)
    if ranks.empty:
        return

    mean_ranks = ranks.groupby("loss")["rank"].mean().sort_values()
    ordered = _present_losses(mean_ranks.index)
    mean_ranks = mean_ranks.loc[ordered]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3" if l.startswith("OLL") else "#9E9E9E" for l in mean_ranks.index]
    ax.barh(mean_ranks.index[::-1], mean_ranks.values[::-1], color=colors[::-1])
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("Average Rank Across Datasets and Metrics")
    ax.axvline(mean_ranks.mean(), color="red", linestyle="--", label="mean rank")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "average_rank.pdf"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate comparison figures from ordinal classification results."
    )
    parser.add_argument(
        "--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR,
        help="Path to the directory containing per-dataset metric folders.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASET_ORDER,
        help="Datasets to include.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ANALYSIS_METRICS,
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--sort_metric", default="off-by-1-accuracy",
        help="Metric used to select the best lr per (dataset, loss) (default: off-by-1-accuracy).",
    )
    parser.add_argument(
        "--lower_is_better", action="store_true",
        help="Sort metric is better when lower (e.g. mae, mse).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--no_heatmap", action="store_true", help="Skip heatmap plots.")
    parser.add_argument("--no_bar", action="store_true", help="Skip bar plots.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.metrics_dir.exists():
        print(f"ERROR: {args.metrics_dir} not found. Run inference.py first.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.metrics_dir, args.datasets)
    df = extract_run_info(df)
    agg = compute_mean_std(df, args.metrics)
    best = select_best_lr(agg, args.sort_metric, args.lower_is_better)

    print(f"Generating figures for metrics: {args.metrics}")

    for metric in args.metrics:
        if f"{metric}_mean" not in best.columns:
            print(f"  Skipping '{metric}' — not in results.")
            continue
        if not args.no_bar:
            grouped_bar(best, metric, args.datasets, args.output_dir)
        if not args.no_heatmap:
            heatmap(best, metric, args.datasets, args.output_dir)

    rank_plot(best, args.metrics, args.datasets, args.output_dir)
    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()