"""
Summarize test-set metrics from inference.py into Excel files.

Produces (under results/):
  - raw_scores.xlsx        : one row per (dataset, loss, seed)
  - mean_std_scores.xlsx   : mean ± std per (dataset, loss) across seeds
  - ranks.xlsx             : rank of each loss per dataset per metric
  - mean_ranks.xlsx        : mean rank of each loss across datasets

Usage:
    python -m scripts.analyze_results                     # all datasets
    python -m scripts.analyze_results --datasets sst5 yelp
    python -m scripts.analyze_results --output_dir results
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np

ROOT_PATH = Path(__file__).parent.parent
DEFAULT_METRICS_DIR = ROOT_PATH / "src" / "outputs_training" / "output_metrics"
DEFAULT_OUTPUT_DIR = ROOT_PATH / "results"

ALL_DATASETS = ["snli", "sst5", "yelp", "amazon_reviews"]

LOSS_ORDER = ["CE", "BCE", "OLL1", "OLL15", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]

ANALYSIS_METRICS = ["off-by-1-accuracy", "off-by-2-accuracy", "mae", "mse", "kendalltau"]

# Metrics where lower = better (used for ranking direction)
LOWER_IS_BETTER = {"mae", "mse"}

PREFIX_COLS = ["dataset", "loss", "pretrained_model", "trained_model"]


def load_metrics(metrics_dir: Path, datasets: list) -> pd.DataFrame:
    """Load per-dataset metrics_test_set.csv files and concatenate them."""
    frames = []
    for ds in datasets:
        csv_path = metrics_dir / ds / "metrics_test_set.csv"
        if not csv_path.is_file():
            print(f"  Warning: no metrics file found for dataset '{ds}' at {csv_path}, skipping.")
            continue
        df = pd.read_csv(csv_path, header=0)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No metrics_test_set.csv files found under {metrics_dir}. "
            "Run inference.py first."
        )

    return pd.concat(frames, ignore_index=True)


def extract_run_info(df: pd.DataFrame) -> pd.DataFrame:
    """Parse learning-rate, epochs, and seed from the trained_model path."""
    # trained_model format: {base_model}-{dataset}-{loss}-{seed}_{epochs}_ep_{lr}_lr_{batch}_batch
    def _parse(row):
        path = str(row["trained_model"])
        parts = path.split("_")
        try:
            lr_idx = parts.index("lr")
            lr = float(parts[lr_idx - 1])
        except (ValueError, IndexError):
            lr = float("nan")
        try:
            ep_idx = parts.index("ep")
            epochs = int(parts[ep_idx - 1])
        except (ValueError, IndexError):
            epochs = -1

        # Seed is encoded as -{loss}-{seed}_ in the path
        seed = -1
        loss = str(row.get("loss", ""))
        pattern = rf"-{re.escape(loss)}-(\d+)_"
        m = re.search(pattern, path)
        if m:
            seed = int(m.group(1))

        return pd.Series({"lr": lr, "epochs": epochs, "seed": seed})

    extra = df.apply(_parse, axis=1)
    return pd.concat([df, extra], axis=1)


def _reorder_losses(df: pd.DataFrame, index_col: str = "loss") -> pd.DataFrame:
    """Reorder rows so losses follow LOSS_ORDER."""
    present = [l for l in LOSS_ORDER if l in df[index_col].values]
    extra = [l for l in df[index_col].unique() if l not in LOSS_ORDER]
    order = present + extra
    df["_order"] = df[index_col].map({l: i for i, l in enumerate(order)})
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Raw per-seed scores
# ---------------------------------------------------------------------------

def build_raw_excel(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Dump one sheet per dataset with raw per-seed scores."""
    out_path = output_dir / "raw_scores.xlsx"
    keep = ["dataset", "loss", "seed"] + [m for m in metrics if m in df.columns]
    sub = df[keep].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # All-datasets combined sheet
        all_df = _reorder_losses(sub.copy())
        all_df.to_excel(writer, sheet_name="all", index=False)

        for ds in sub["dataset"].unique():
            sheet = sub[sub["dataset"] == ds].drop(columns="dataset").copy()
            sheet = _reorder_losses(sheet)
            sheet.to_excel(writer, sheet_name=ds, index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# 2. Mean ± std per (dataset, loss)
# ---------------------------------------------------------------------------

def compute_mean_std(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Compute mean and std of each metric grouped by (dataset, loss)."""
    metric_cols = [m for m in metrics if m in df.columns]
    agg = (
        df.groupby(["dataset", "loss"])[metric_cols]
        .agg(["mean", "std"])
    )
    # Flatten MultiIndex columns: (metric, stat) → metric_mean / metric_std
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def build_mean_std_excel(df: pd.DataFrame, metrics: list, output_dir: Path) -> pd.DataFrame:
    """Dump mean+std per (dataset, loss) into Excel; return the aggregated DataFrame."""
    agg = compute_mean_std(df, metrics)
    out_path = output_dir / "mean_std_scores.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_df = _reorder_losses(agg.copy())
        all_df.to_excel(writer, sheet_name="all", index=False)

        for ds in agg["dataset"].unique():
            sheet = agg[agg["dataset"] == ds].drop(columns="dataset").copy()
            sheet = _reorder_losses(sheet)
            sheet.to_excel(writer, sheet_name=ds, index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")
    return agg


# ---------------------------------------------------------------------------
# 3. Ranks per (dataset, metric)
# ---------------------------------------------------------------------------

def compute_ranks(agg: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Rank each loss per dataset per metric based on mean score."""
    frames = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in agg.columns:
            continue
        ascending = metric in LOWER_IS_BETTER
        ranked = agg[["dataset", "loss", mean_col]].copy()
        ranked["rank"] = (
            ranked.groupby("dataset")[mean_col]
            .rank(ascending=ascending, method="min")
        )
        ranked["metric"] = metric
        ranked = ranked.rename(columns={mean_col: "mean_score"})
        frames.append(ranked)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_ranks_excel(agg: pd.DataFrame, metrics: list, datasets: list, output_dir: Path) -> pd.DataFrame:
    """Dump per-metric ranks (loss × dataset) into Excel; return rank DataFrame."""
    ranks = compute_ranks(agg, metrics)
    if ranks.empty:
        return ranks

    out_path = output_dir / "ranks.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # One sheet per metric showing loss × dataset rank pivot
        for metric in metrics:
            sub = ranks[ranks["metric"] == metric]
            if sub.empty:
                continue
            pivot = sub.pivot(index="loss", columns="dataset", values="rank")
            present_losses = [l for l in LOSS_ORDER if l in pivot.index]
            extra = [l for l in pivot.index if l not in LOSS_ORDER]
            present_datasets = [d for d in datasets if d in pivot.columns]
            pivot = pivot.loc[present_losses + extra, present_datasets]
            pivot.to_excel(writer, sheet_name=metric[:31])

        # Summary sheet: all ranks combined
        summary = _reorder_losses(ranks.copy())
        summary.to_excel(writer, sheet_name="all_ranks", index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")
    return ranks


# ---------------------------------------------------------------------------
# 4. Mean rank across datasets
# ---------------------------------------------------------------------------

def build_mean_ranks_excel(ranks: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Compute and dump mean rank of each loss across datasets per metric."""
    if ranks.empty:
        return

    out_path = output_dir / "mean_ranks.xlsx"

    # Per-metric mean rank across datasets
    mean_per_metric = (
        ranks.groupby(["loss", "metric"])["rank"]
        .mean()
        .rename("mean_rank")
        .reset_index()
    )

    # Overall mean rank across all metrics and datasets
    overall = (
        ranks.groupby("loss")["rank"]
        .mean()
        .rename("mean_rank")
        .reset_index()
    )
    overall["metric"] = "overall"

    combined = pd.concat([mean_per_metric, overall], ignore_index=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Pivot: loss × metric
        pivot = combined.pivot(index="loss", columns="metric", values="mean_rank")
        present_losses = [l for l in LOSS_ORDER if l in pivot.index]
        extra = [l for l in pivot.index if l not in LOSS_ORDER]
        # Put overall last
        metric_cols = [m for m in metrics if m in pivot.columns] + (
            ["overall"] if "overall" in pivot.columns else []
        )
        pivot = pivot.loc[present_losses + extra, metric_cols]
        pivot.to_excel(writer, sheet_name="mean_ranks")

        # Detailed sheet
        detail = _reorder_losses(combined.copy())
        detail.to_excel(writer, sheet_name="detail", index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# Legacy helpers (used by visualize_results.py)
# ---------------------------------------------------------------------------

def best_per_loss(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """For each (dataset, loss) pair keep only the run with the best metric value."""
    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found. Available: "
            f"{[c for c in df.columns if c not in PREFIX_COLS]}"
        )
    agg = df.groupby(["dataset", "loss"])[metric]
    idx = agg.idxmax() if higher_is_better else agg.idxmin()
    return df.loc[idx.dropna().astype(int)].reset_index(drop=True)


def pivot_table(df: pd.DataFrame, metric: str, datasets: list) -> pd.DataFrame:
    """Return a (loss × dataset) pivot table for a given metric."""
    sub = df[df["dataset"].isin(datasets)][["dataset", "loss", metric]].copy()
    sub[metric] = sub[metric].round(4)
    pivot = sub.pivot(index="loss", columns="dataset", values=metric)
    present = [l for l in LOSS_ORDER if l in pivot.index]
    extra = [l for l in pivot.index if l not in LOSS_ORDER]
    return pivot.loc[present + extra]


def print_table(pivot: pd.DataFrame, metric: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Metric: {metric}")
    print(f"{'='*60}")
    print(pivot.to_string())
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate inference results into Excel files (raw, mean/std, ranks)."
    )
    parser.add_argument(
        "--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR,
        help="Directory containing per-dataset metric folders.",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS,
        help="Datasets to include.",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ANALYSIS_METRICS,
        help="Metrics to aggregate and rank.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write Excel files (default: results/).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading metrics from {args.metrics_dir} ...")
    df = load_metrics(args.metrics_dir, args.datasets)
    df = extract_run_info(df)

    print(f"  Total rows : {len(df)}")
    print(f"  Datasets   : {sorted(df['dataset'].unique())}")
    print(f"  Losses     : {sorted(df['loss'].unique())}")
    seeds = sorted(df['seed'].unique())
    print(f"  Seeds      : {seeds}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding Excel outputs ...")
    build_raw_excel(df, args.metrics, args.output_dir)
    agg = build_mean_std_excel(df, args.metrics, args.output_dir)
    ranks = build_ranks_excel(agg, args.metrics, args.datasets, args.output_dir)
    build_mean_ranks_excel(ranks, args.metrics, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
