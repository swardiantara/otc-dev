"""
Summarize test-set metrics from inference.py into Excel files.

Pipeline
--------
1. Parse seed and learning-rate from the trained_model path.
2. Group by (dataset, loss, lr) → mean & std across the five seeds.
3. Select the best lr per (dataset, loss) via a configurable reference metric.
4. Use the best-lr mean scores for ranking and downstream analysis.

Outputs (under results/ by default)
------------------------------------
  raw_scores.xlsx       – one row per (dataset, loss, seed, lr)
  mean_std_scores.xlsx  – mean ± std per (dataset, loss, lr) across seeds
  best_lr_scores.xlsx   – best-lr mean scores per (dataset, loss)
  ranks.xlsx            – rank of each loss per dataset per metric
  mean_ranks.xlsx       – mean rank of each loss across datasets

Usage
-----
    python -m scripts.analyze_results
    python -m scripts.analyze_results --datasets sst5 yelp
    python -m scripts.analyze_results --sort_metric mae --lower_is_better
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


# ---------------------------------------------------------------------------
# Loading & parsing
# ---------------------------------------------------------------------------

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
    """Parse lr, epochs, and seed from the trained_model path.

    Path format:
        {base_model}-{dataset}-{loss}-{seed}_{epochs}_ep_{lr}_lr_{batch}_batch
    """
    def _parse(row):
        path = str(row["trained_model"])
        parts = path.split("_")

        # Learning rate
        try:
            lr_idx = parts.index("lr")
            lr = float(parts[lr_idx - 1])
        except (ValueError, IndexError):
            lr = float("nan")

        # Epochs
        try:
            ep_idx = parts.index("ep")
            epochs = int(parts[ep_idx - 1])
        except (ValueError, IndexError):
            epochs = -1

        # Seed: -{loss}-{seed}_ in the path
        seed = -1
        loss = str(row.get("loss", ""))
        m = re.search(rf"-{re.escape(loss)}-(\d+)_", path)
        if m:
            seed = int(m.group(1))

        return pd.Series({"lr": lr, "epochs": epochs, "seed": seed})

    extra = df.apply(_parse, axis=1)
    return pd.concat([df, extra], axis=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reorder_losses(df: pd.DataFrame, index_col: str = "loss") -> pd.DataFrame:
    """Sort rows so losses follow LOSS_ORDER."""
    present = [l for l in LOSS_ORDER if l in df[index_col].values]
    extra = [l for l in df[index_col].unique() if l not in LOSS_ORDER]
    order_map = {l: i for i, l in enumerate(present + extra)}
    df = df.copy()
    df["_order"] = df[index_col].map(order_map)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _present_losses(index):
    present = [l for l in LOSS_ORDER if l in index]
    extra = [l for l in index if l not in LOSS_ORDER]
    return present + extra


# ---------------------------------------------------------------------------
# 1. Raw per-seed Excel
# ---------------------------------------------------------------------------

def build_raw_excel(df: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """One row per (dataset, loss, lr, seed). One sheet per dataset + combined."""
    out_path = output_dir / "raw_scores.xlsx"
    keep = ["dataset", "loss", "lr", "seed"] + [m for m in metrics if m in df.columns]
    sub = df[keep].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _reorder_losses(sub).to_excel(writer, sheet_name="all", index=False)
        for ds in sub["dataset"].unique():
            sheet = _reorder_losses(sub[sub["dataset"] == ds].drop(columns="dataset").copy())
            sheet.to_excel(writer, sheet_name=str(ds), index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# 2. Mean ± std per (dataset, loss, lr) — aggregated across seeds
# ---------------------------------------------------------------------------

def compute_mean_std(df: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Mean and std of each metric grouped by (dataset, loss, lr)."""
    metric_cols = [m for m in metrics if m in df.columns]
    agg = (
        df.groupby(["dataset", "loss", "lr"])[metric_cols]
        .agg(["mean", "std"])
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def build_mean_std_excel(df: pd.DataFrame, metrics: list, output_dir: Path) -> pd.DataFrame:
    """Dump mean+std per (dataset, loss, lr) into Excel. Returns the aggregated DataFrame."""
    agg = compute_mean_std(df, metrics)
    out_path = output_dir / "mean_std_scores.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _reorder_losses(agg).to_excel(writer, sheet_name="all", index=False)
        for ds in agg["dataset"].unique():
            sheet = _reorder_losses(agg[agg["dataset"] == ds].drop(columns="dataset").copy())
            sheet.to_excel(writer, sheet_name=str(ds), index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")
    return agg


# ---------------------------------------------------------------------------
# 3. Best-LR selection per (dataset, loss)
# ---------------------------------------------------------------------------

def select_best_lr(agg: pd.DataFrame, sort_metric: str, lower_is_better: bool) -> pd.DataFrame:
    """For each (dataset, loss) pick the lr with the best mean score on sort_metric.

    Returns a DataFrame with one row per (dataset, loss) containing the mean
    scores of the best lr.
    """
    mean_col = f"{sort_metric}_mean"
    if mean_col not in agg.columns:
        available = [c for c in agg.columns if c.endswith("_mean")]
        raise ValueError(
            f"Sort metric column '{mean_col}' not found. "
            f"Available mean columns: {available}"
        )

    if lower_is_better:
        idx = agg.groupby(["dataset", "loss"])[mean_col].idxmin()
    else:
        idx = agg.groupby(["dataset", "loss"])[mean_col].idxmax()

    return agg.loc[idx.dropna().astype(int)].reset_index(drop=True)


def build_best_lr_excel(best: pd.DataFrame, datasets: list, output_dir: Path) -> None:
    """Dump best-lr mean scores (one row per dataset-loss) into Excel."""
    out_path = output_dir / "best_lr_scores.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _reorder_losses(best).to_excel(writer, sheet_name="all", index=False)
        for ds in best["dataset"].unique():
            sheet = _reorder_losses(best[best["dataset"] == ds].drop(columns="dataset").copy())
            sheet.to_excel(writer, sheet_name=str(ds), index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# 4. Ranks per (dataset, metric) — based on best-lr mean scores
# ---------------------------------------------------------------------------

def compute_ranks(best: pd.DataFrame, metrics: list) -> pd.DataFrame:
    """Rank each loss per dataset per metric using best-lr mean scores."""
    frames = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        if mean_col not in best.columns:
            continue
        ascending = metric in LOWER_IS_BETTER
        sub = best[["dataset", "loss", mean_col]].copy()
        sub["rank"] = sub.groupby("dataset")[mean_col].rank(ascending=ascending, method="min")
        sub["metric"] = metric
        sub = sub.rename(columns={mean_col: "mean_score"})
        frames.append(sub)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_ranks_excel(best: pd.DataFrame, metrics: list, datasets: list, output_dir: Path) -> pd.DataFrame:
    """Dump per-metric rank pivots (loss × dataset) into Excel. Returns rank DataFrame."""
    ranks = compute_ranks(best, metrics)
    if ranks.empty:
        return ranks

    out_path = output_dir / "ranks.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for metric in metrics:
            sub = ranks[ranks["metric"] == metric]
            if sub.empty:
                continue
            pivot = sub.pivot(index="loss", columns="dataset", values="rank")
            present_ds = [d for d in datasets if d in pivot.columns]
            pivot = pivot.loc[_present_losses(pivot.index), present_ds]
            pivot.to_excel(writer, sheet_name=metric[:31])

        _reorder_losses(ranks).to_excel(writer, sheet_name="all_ranks", index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")
    return ranks


# ---------------------------------------------------------------------------
# 5. Mean rank across datasets
# ---------------------------------------------------------------------------

def build_mean_ranks_excel(ranks: pd.DataFrame, metrics: list, output_dir: Path) -> None:
    """Mean rank of each loss across datasets, per metric and overall."""
    if ranks.empty:
        return

    out_path = output_dir / "mean_ranks.xlsx"

    per_metric = (
        ranks.groupby(["loss", "metric"])["rank"]
        .mean()
        .rename("mean_rank")
        .reset_index()
    )
    overall = (
        ranks.groupby("loss")["rank"]
        .mean()
        .rename("mean_rank")
        .reset_index()
    )
    overall["metric"] = "overall"

    combined = pd.concat([per_metric, overall], ignore_index=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pivot = combined.pivot(index="loss", columns="metric", values="mean_rank")
        metric_cols = [m for m in metrics if m in pivot.columns]
        if "overall" in pivot.columns:
            metric_cols = metric_cols + ["overall"]
        pivot = pivot.loc[_present_losses(pivot.index), metric_cols]
        pivot.to_excel(writer, sheet_name="mean_ranks")
        _reorder_losses(combined).to_excel(writer, sheet_name="detail", index=False)

    print(f"  Saved {out_path.relative_to(ROOT_PATH)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate inference results into Excel files.\n\n"
            "Pipeline: parse seed+lr → mean/std per (dataset,loss,lr) across seeds "
            "→ select best lr per (dataset,loss) → rank losses."
        )
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
        "--sort_metric", default="off-by-1-accuracy",
        help="Metric used to select the best lr per (dataset, loss) (default: off-by-1-accuracy).",
    )
    parser.add_argument(
        "--lower_is_better", action="store_true",
        help="If set, the best lr is chosen by the LOWEST sort_metric value (e.g. mae, mse).",
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
    print(f"  Seeds      : {sorted(df['seed'].unique())}")
    print(f"  LRs        : {sorted(df['lr'].unique())}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nBuilding Excel outputs ...")
    build_raw_excel(df, args.metrics, args.output_dir)

    agg = build_mean_std_excel(df, args.metrics, args.output_dir)

    best = select_best_lr(agg, args.sort_metric, args.lower_is_better)
    build_best_lr_excel(best, args.datasets, args.output_dir)

    ranks = build_ranks_excel(best, args.metrics, args.datasets, args.output_dir)
    build_mean_ranks_excel(ranks, args.metrics, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
