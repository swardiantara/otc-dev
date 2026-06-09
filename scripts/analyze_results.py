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
DEFAULT_METRICS_DIR = ROOT_PATH / "src" / "outputs_training" / "output_metrics"

ALL_DATASETS = ["snli", "sst5", "yelp", "amazon_reviews"]

LOSS_ORDER = ["CE", "BCE", "OLL1", "OLL15", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]

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
    """Parse learning-rate and epochs from the trained_model path."""
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
        "--metrics_dir", type=Path, default=DEFAULT_METRICS_DIR,
        help="Directory containing per-dataset metric folders (default: src/outputs_training/output_metrics).",
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
        "--output", type=Path, default="results/summary.csv",
        help="If provided, save the full best-run table to this CSV path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading metrics from {args.metrics_dir} ...")
    df = load_metrics(args.metrics_dir, args.datasets)
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
