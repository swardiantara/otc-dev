"""Aggregate CPL embedding fine-tuning results and pick the best variant.

Scans results/cpl_embedding/{dataset}/*.csv (one file per run, written by
src/finetune_cpl.py), parses the CPL variant encoded in each filename, writes a
summary, and selects the best embedding per dataset by validation kNN-MAE
(``kNN_MAE_avg``, lower is better) into src/cpl_embedding_config.json — the
analogue of src/embedding_config.json for the contrastive flow.

Filename convention (set by finetune_cpl.py):
  {model_alias}-{dataset}-cpl-{constraint}-{metric}.csv
  e.g.  bert-tiny-sst5-cpl-S-B-E.csv
        bert-tiny-snli-cpl-H-L-E.csv
where constraint in {S-P, S-B, H-L, H-S} and metric in {E, C}.

The resulting config is consumed by the classifier stage: build the Hub id
``{hf_user}/{model_alias}-{dataset}-cpl-{constraint}-{metric}`` and pass it as
``--model_checkpoint`` to src/training.py (see scripts/run_cpl.sh).
"""
import csv
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
CPL_DIR = ROOT_PATH / "results" / "cpl_embedding"
CONFIG_PATH = ROOT_PATH / "src" / "cpl_embedding_config.json"

DATASETS = ["sst5", "snli", "amazon_reviews", "yelp"]
CONSTRAINTS = {"S-P", "S-B", "H-L", "H-S"}
METRICS = {"E", "C"}

HYPER_COLS = ["dataset", "model_alias", "constraint", "metric"]
SELECTION_METRIC = "kNN_MAE_avg"  # lower is better


def parse_model_id(stem, dataset):
    """Return {model_alias, dataset, constraint, metric} from a result filename."""
    marker = f"-{dataset}-cpl-"
    idx = stem.find(marker)
    if idx < 0:
        return None
    model_alias = stem[:idx]
    rest = stem[idx + len(marker):]  # e.g. "S-B-E" or "H-L-E"

    parts = rest.split("-")
    if len(parts) < 2:
        return None
    metric = parts[-1]
    constraint = "-".join(parts[:-1])  # constraints contain a hyphen (e.g. S-B)

    if constraint not in CONSTRAINTS or metric not in METRICS:
        return None

    return {
        "model_alias": model_alias,
        "dataset": dataset,
        "constraint": constraint,
        "metric": metric,
    }


def collect_rows():
    rows, skipped = [], []
    for dataset in DATASETS:
        dataset_dir = CPL_DIR / dataset
        if not dataset_dir.exists():
            continue
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            if csv_path.stem == "summary" or "_tsne" in csv_path.stem:
                continue
            parsed = parse_model_id(csv_path.stem, dataset)
            if parsed is None:
                skipped.append(csv_path.name)
                continue
            with open(csv_path, newline="") as f:
                for metric_row in csv.DictReader(f):
                    rows.append({**parsed, **metric_row})
    if skipped:
        print(f"Warning: could not parse {len(skipped)} file(s): {skipped}", file=sys.stderr)
    return rows


def write_summary(rows):
    metric_cols = sorted({k for row in rows for k in row if k not in HYPER_COLS})
    fieldnames = HYPER_COLS + metric_cols
    out_path = CPL_DIR / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary written -> {out_path}  ({len(rows)} run(s))")
    return out_path


def select_best(rows):
    """Best (lowest kNN_MAE_avg) variant per dataset."""
    best = {}
    for row in rows:
        try:
            score = float(row[SELECTION_METRIC])
        except (KeyError, ValueError, TypeError):
            continue
        ds = row["dataset"]
        if ds not in best or score < best[ds]["_score"]:
            best[ds] = {
                "model_alias": row["model_alias"],
                "constraint": row["constraint"],
                "metric": row["metric"],
                "model_id": f"{row['model_alias']}-{ds}-cpl-{row['constraint']}-{row['metric']}",
                "_score": score,
            }
    return best


def main():
    rows = collect_rows()
    if not rows:
        print(f"No results found under {CPL_DIR}. Run src/finetune_cpl.py first.")
        return

    write_summary(rows)
    best = select_best(rows)

    # Print a compact per-dataset preview.
    print()
    for dataset in DATASETS:
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        print(f"--- {dataset} ---")
        print(f"  {'constraint':10s} {'metric':6s} {'NOS_avg':>8s} {'kNN_MAE_avg':>11s}")
        for r in sorted(ds_rows, key=lambda r: r.get(SELECTION_METRIC, "inf")):
            nos = r.get("NOS_avg", "n/a")
            mae = r.get(SELECTION_METRIC, "n/a")
            try:
                nos = f"{float(nos):.4f}"
            except (ValueError, TypeError):
                pass
            try:
                mae = f"{float(mae):.4f}"
            except (ValueError, TypeError):
                pass
            star = "  <- best" if dataset in best and r["constraint"] == best[dataset]["constraint"] \
                and r["metric"] == best[dataset]["metric"] else ""
            print(f"  {r['constraint']:10s} {r['metric']:6s} {nos:>8s} {mae:>11s}{star}")
        print()

    # Write the selection config (drop the private score field).
    config = {ds: {k: v for k, v in info.items() if not k.startswith("_")}
              for ds, info in best.items()}
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Best-variant config written -> {CONFIG_PATH}")


if __name__ == "__main__":
    main()
