"""
Generate comparison plots from test-set metrics.

Produces (saved under results/figures/):
  - bar_<metric>.png      : grouped bar chart per dataset for each metric
  - heatmap_<metric>.png  : loss × dataset heatmap for each metric

Usage:
    python -m scripts.visualize_results
    python -m scripts.visualize_results --metrics accuracy mae --datasets sst5 yelp
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

# Re-use helpers from analyze_results
from scripts.analyze_results import (
    load_metrics, extract_run_info, best_per_loss, LOSS_ORDER,
    DEFAULT_METRICS_CSV,
)

ROOT_PATH = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = ROOT_PATH / "results" / "figures"

# "amazon_reviews" excluded from defaults; kept in list so plots work if re-added later.
DATASET_ORDER = ["sst5", "yelp", "snli"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_score": "Weighted F1",
    "mae": "MAE",
    "mse": "MSE",
    "kendalltau": "Kendall's τ",
    "off-by-1-accuracy": "Off-by-1 Accuracy",
}

# Metrics where lower = better (affects heatmap colour direction)
LOWER_IS_BETTER = {"mae", "mse"}


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def grouped_bar(best: pd.DataFrame, metric: str, datasets: list, output_dir: Path) -> None:
    """One group of bars per dataset, one bar per loss function."""
    sub = best[best["dataset"].isin(datasets)][["dataset", "loss", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    pivot = sub.pivot(index="loss", columns="dataset", values=metric)

    present_losses = [l for l in LOSS_ORDER if l in pivot.index]
    present_datasets = [d for d in DATASET_ORDER if d in pivot.columns]
    pivot = pivot.loc[present_losses, present_datasets]

    n_losses = len(present_losses)
    n_datasets = len(present_datasets)
    x = np.arange(n_losses)
    width = 0.8 / n_datasets
    palette = sns.color_palette("tab10", n_datasets)

    fig, ax = plt.subplots(figsize=(max(10, n_losses * 1.2), 5))

    for i, dataset in enumerate(present_datasets):
        offset = (i - n_datasets / 2 + 0.5) * width
        vals = pivot[dataset].values
        bars = ax.bar(x + offset, vals, width * 0.9, label=dataset, color=palette[i])

    ax.set_xticks(x)
    ax.set_xticklabels(present_losses, fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} by Loss Function and Dataset")
    ax.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()

    out_path = output_dir / f"bar_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


def heatmap(best: pd.DataFrame, metric: str, datasets: list, output_dir: Path) -> None:
    """Loss × Dataset heatmap."""
    sub = best[best["dataset"].isin(datasets)][["dataset", "loss", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    pivot = sub.pivot(index="loss", columns="dataset", values=metric)

    present_losses = [l for l in LOSS_ORDER if l in pivot.index]
    present_datasets = [d for d in DATASET_ORDER if d in pivot.columns]
    pivot = pivot.loc[present_losses, present_datasets].astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(present_datasets) * 1.8), max(4, len(present_losses) * 0.7)))

    cmap = "RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn"
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=True, fmt=".3f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": METRIC_LABELS.get(metric, metric)},
    )
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — Loss × Dataset")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Loss Function")
    fig.tight_layout()

    out_path = output_dir / f"heatmap_{metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


def rank_plot(best: pd.DataFrame, metrics: list, datasets: list, output_dir: Path) -> None:
    """Average rank of each loss across datasets and metrics (lower = better overall)."""
    sub = best[best["dataset"].isin(datasets)].copy()

    rank_frames = []
    for metric in metrics:
        if metric not in sub.columns:
            continue
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        ascending = metric in LOWER_IS_BETTER
        ranked = (
            sub.groupby("dataset")[["loss", metric]]
            .apply(lambda g: g.set_index("loss")[metric].rank(ascending=ascending))
            .reset_index()
            .melt(id_vars="dataset", var_name="loss", value_name="rank")
        )
        ranked["metric"] = metric
        rank_frames.append(ranked)

    if not rank_frames:
        return

    all_ranks = pd.concat(rank_frames)
    mean_ranks = all_ranks.groupby("loss")["rank"].mean().sort_values()

    present_losses = [l for l in LOSS_ORDER if l in mean_ranks.index]
    extra = [l for l in mean_ranks.index if l not in LOSS_ORDER]
    mean_ranks = mean_ranks.loc[present_losses + extra]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3" if l.startswith("OLL") else "#9E9E9E" for l in mean_ranks.index]
    ax.barh(mean_ranks.index[::-1], mean_ranks.values[::-1], color=colors[::-1])
    ax.set_xlabel("Average Rank (lower = better)")
    ax.set_title("Average Rank Across Datasets and Metrics")
    ax.axvline(mean_ranks.mean(), color="red", linestyle="--", label="mean rank")
    ax.legend()
    fig.tight_layout()

    out_path = output_dir / "average_rank.png"
    fig.savefig(out_path, dpi=150)
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
        "--metrics_csv", type=Path, default=DEFAULT_METRICS_CSV,
        help="Path to metrics_test_set.csv.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DATASET_ORDER,
        help="Datasets to include.",
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["accuracy", "mae", "mse", "kendalltau", "off-by-1-accuracy"],
        help="Metrics to plot.",
    )
    parser.add_argument(
        "--sort_metric", default="accuracy",
        help="Metric used to pick best run per (dataset, loss) (default: accuracy).",
    )
    parser.add_argument(
        "--lower_is_better", action="store_true",
        help="Sort metric is better when lower (e.g. mae).",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for saved figures (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no_heatmap", action="store_true", help="Skip heatmap plots.",
    )
    parser.add_argument(
        "--no_bar", action="store_true", help="Skip bar plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.metrics_csv.is_file():
        print(f"ERROR: {args.metrics_csv} not found. Run inference.py first.")
        raise SystemExit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.metrics_csv)
    df = extract_run_info(df)
    higher = not args.lower_is_better
    best = best_per_loss(df, args.sort_metric, higher_is_better=higher)

    print(f"Generating figures for metrics: {args.metrics}")

    for metric in args.metrics:
        if metric not in best.columns:
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
