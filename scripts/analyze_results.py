"""
Summarize test-set metrics from inference.py into readable tables.

Usage:
    python -m scripts.analyze_results                     # all datasets
    python -m scripts.analyze_results --datasets sst5 yelp
    python -m scripts.analyze_results --metric mae        # sort by MAE
    python -m scripts.analyze_results --output results/summary.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

ROOT_PATH = Path(__file__).parent.parent
DEFAULT_METRICS_CSV = (
    ROOT_PATH / "src" / "outputs_training" / "output_metrics" / "metrics_test_set.csv"
)

METRIC_COLS = [
    "accuracy", "precision", "recall", "f1_score",
    "distance_1", "distance_2", "distance_3", "distance_4",
    "off-by-1-accuracy", "off-by-2-accuracy", "off-by-3-accuracy",
    "mae", "mse", "kendalltau",
]

# Columns written by inference.py before metric columns
PREFIX_COLS = ["dataset", "loss", "pretrained_model", "trained_model"]

LOSS_ORDER = ["CE", "OLL1", "OLL15", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """Load the raw metrics CSV and assign column names."""
    raw = pd.read_csv(csv_path, header=None)

    n_prefix = len(PREFIX_COLS)
    n_metrics = raw.shape[1] - n_prefix
    auto_metric_cols = [f"metric_{i}" for i in range(n_metrics)]

    raw.columns = PREFIX_COLS + auto_metric_cols

    # Try to assign known metric names to the first columns after prefix
    named_metrics = METRIC_COLS[:n_metrics]
    rename = {f"metric_{i}": named_metrics[i] for i in range(len(named_metrics))}
    raw = raw.rename(columns=rename)

    return raw


def extract_run_info(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dataset, loss, learning-rate and seed from the trained_model path."""
    # trained_model format: google/bert_uncased_L-2_H-128_A-2-{dataset}-{loss}-{seed}_{epochs}_ep_{lr}_lr_{batch}_batch
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
        return pd.Series({"lr": lr, "epochs": epochs})

    extra = df.apply(_parse, axis=1)
    return pd.concat([df, extra], axis=1)


def best_per_loss(df: pd.DataFrame, metric: str, higher_is_better: bool) -> pd.DataFrame:
    """For each (dataset, loss) pair keep only the run with the best metric value."""
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {[c for c in df.columns if c not in PREFIX_COLS]}")

    agg = df.groupby(["dataset", "loss"])[metric]
    if higher_is_better:
        idx = agg.idxmax()
    else:
        idx = agg.idxmin()

    return df.loc[idx.dropna().astype(int)].reset_index(drop=True)


def pivot_table(df: pd.DataFrame, metric: str, datasets: list) -> pd.DataFrame:
    """Return a (loss × dataset) pivot table for a given metric."""
    sub = df[df["dataset"].isin(datasets)][["dataset", "loss", metric]].copy()
    sub[metric] = sub[metric].round(4)
    pivot = sub.pivot(index="loss", columns="dataset", values=metric)

    # Reorder rows to match paper loss ordering
    present = [l for l in LOSS_ORDER if l in pivot.index]
    extra = [l for l in pivot.index if l not in LOSS_ORDER]
    pivot = pivot.loc[present + extra]

    return pivot


def print_table(pivot: pd.DataFrame, metric: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Metric: {metric}")
    print(f"{'='*60}")
    print(pivot.to_string())
    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize inference results into pivot tables by dataset and loss function."
    )
    parser.add_argument(
        "--metrics_csv", type=Path, default=DEFAULT_METRICS_CSV,
        help="Path to metrics_test_set.csv produced by inference.py.",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        # "amazon_reviews" excluded: dataset source defunct; re-add when available.
        default=["snli", "sst5", "yelp"],
        help="Datasets to include in summary.",
    )
    parser.add_argument(
        "--metrics", nargs="+",
        default=["accuracy", "mae", "mse", "kendalltau", "off-by-1-accuracy"],
        help="Metrics to print tables for.",
    )
    parser.add_argument(
        "--sort_metric", default="accuracy",
        help="Metric used to select best run per (dataset, loss) pair (default: accuracy).",
    )
    parser.add_argument(
        "--lower_is_better", action="store_true",
        help="If set, the best run is the one with the LOWEST sort_metric (e.g. mae, mse).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="If provided, save the full best-run table to this CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.metrics_csv.is_file():
        print(f"ERROR: metrics file not found at {args.metrics_csv}")
        print("Run inference.py first to generate evaluation results.")
        raise SystemExit(1)

    print(f"Loading {args.metrics_csv} ...")
    df = load_metrics(args.metrics_csv)
    df = extract_run_info(df)

    print(f"  Total rows: {len(df)}")
    print(f"  Datasets  : {sorted(df['dataset'].unique())}")
    print(f"  Losses    : {sorted(df['loss'].unique())}")

    higher = not args.lower_is_better
    best = best_per_loss(df, args.sort_metric, higher_is_better=higher)

    for metric in args.metrics:
        if metric not in best.columns:
            print(f"  Warning: metric '{metric}' not found in results, skipping.")
            continue
        pivot = pivot_table(best, metric, args.datasets)
        print_table(pivot, metric)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        best.to_csv(args.output, index=False)
        print(f"Saved best-run table to {args.output}")


if __name__ == "__main__":
    main()
