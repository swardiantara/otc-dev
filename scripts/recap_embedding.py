"""Aggregate embedding fine-tuning results into a single summary CSV.

Scans results/embedding/{dataset}/*.csv (one file per run), parses the
hyperparameters encoded in each filename, and writes results/embedding/summary.csv.

Filename convention (set by finetune_embedding.py):
  {model_alias}-{dataset}-{k_or_full}-{margin_type}-{distance_metric}.csv
  e.g.  bert-tiny-sst5-k1-fixed-cosine.csv
        bert-tiny-sst5-full-adaptive-euclidean.csv
"""
import csv
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
EMBEDDING_DIR = ROOT_PATH / "results" / "embedding"
DATASETS = ["sst5", "snli", "amazon_reviews", "yelp"]
MARGIN_TYPES = {"fixed", "adaptive"}
DISTANCE_METRICS = {"cosine", "euclidean"}

HYPER_COLS = ["dataset", "model_alias", "pair_mode", "k_proxies",
              "margin_type", "distance_metric"]


def parse_model_id(stem, dataset):
    """Return hyperparameter dict parsed from a result-CSV filename stem."""
    marker = f"-{dataset}-"
    idx = stem.find(marker)
    if idx < 0:
        return None
    model_alias = stem[:idx]
    rest = stem[idx + len(marker):]  # e.g. "k1-fixed-cosine" or "full-adaptive-euclidean"

    # Last two tokens are always margin_type and distance_metric
    parts = rest.split("-")
    if len(parts) < 3:
        return None
    distance_metric = parts[-1]
    margin_type = parts[-2]
    k_part = "-".join(parts[:-2])   # "k1", "k3", "full", …

    if distance_metric not in DISTANCE_METRICS or margin_type not in MARGIN_TYPES:
        return None

    if k_part == "full":
        pair_mode, k_proxies = "full", "full"
    elif k_part.startswith("k") and k_part[1:].isdigit():
        pair_mode, k_proxies = "proxy", int(k_part[1:])
    else:
        return None

    return {
        "model_alias": model_alias,
        "dataset": dataset,
        "pair_mode": pair_mode,
        "k_proxies": k_proxies,
        "margin_type": margin_type,
        "distance_metric": distance_metric,
    }


def main():
    rows = []
    skipped = []

    for dataset in DATASETS:
        dataset_dir = EMBEDDING_DIR / dataset
        if not dataset_dir.exists():
            continue
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            # Skip the summary file itself and any intermediate artefacts
            if csv_path.stem in {"summary"} or "_tsne" in csv_path.stem:
                continue
            parsed = parse_model_id(csv_path.stem, dataset)
            if parsed is None:
                skipped.append(csv_path.name)
                continue
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for metric_row in reader:
                    rows.append({**parsed, **metric_row})

    if skipped:
        print(f"Warning: could not parse {len(skipped)} file(s): {skipped}", file=sys.stderr)

    if not rows:
        print("No results found under results/embedding/.")
        return

    # Consistent column ordering: hyperparams first, then metrics alphabetically
    metric_cols = sorted({k for row in rows for k in row if k not in HYPER_COLS})
    fieldnames = HYPER_COLS + metric_cols

    out_path = EMBEDDING_DIR / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary written → {out_path}  ({len(rows)} run(s))")

    # Print a compact pivot-style preview grouped by dataset
    print()
    cur_dataset = None
    for row in sorted(rows, key=lambda r: (r["dataset"], r["pair_mode"],
                                            str(r["k_proxies"]), r["margin_type"],
                                            r["distance_metric"])):
        if row["dataset"] != cur_dataset:
            cur_dataset = row["dataset"]
            print(f"\n--- {cur_dataset} ---")
            print(f"  {'pair':5s} {'k':>4s} {'margin':8s} {'metric':9s} "
                  f"{'NOS_avg':>8s} {'kNN_MAE_avg':>11s}")
        k = str(row["k_proxies"])
        nos = row.get("NOS_avg", "n/a")
        mae = row.get("kNN_MAE_avg", "n/a")
        try:
            nos = f"{float(nos):.4f}"
        except (ValueError, TypeError):
            pass
        try:
            mae = f"{float(mae):.4f}"
        except (ValueError, TypeError):
            pass
        print(f"  {row['pair_mode']:5s} {k:>4s} {row['margin_type']:8s} "
              f"{row['distance_metric']:9s} {nos:>8s} {mae:>11s}")


if __name__ == "__main__":
    main()
