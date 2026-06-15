"""Print and save class distribution and average word count for every dataset split."""
import csv
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

ROOT_PATH = Path(__file__).parent.parent
SPLITS = ["train", "validation", "test"]


def _load_split(dataset_name, split, cfg):
    data_path = Path(cfg["path"]) / dataset_name / f"{dataset_name}_{split}.csv"
    if not data_path.exists():
        return None, None
    raw = load_dataset("csv", data_files={split: str(data_path)})[split]
    text_col1, text_col2 = cfg["task"]
    if text_col2 is not None:
        texts = [f"{p} {h}" for p, h in zip(raw[text_col1], raw[text_col2])]
    else:
        texts = list(raw[text_col1])
    labels = list(raw["label"])
    valid = [(t, l) for t, l in zip(texts, labels) if l >= 0]
    if not valid:
        return [], []
    texts, labels = zip(*valid)
    return list(texts), list(labels)


def dataset_stats(dataset_name, cfg):
    rows = []
    print(f"\n{'='*64}")
    print(f"  {dataset_name.upper()}   (num_classes={cfg['num_classes']})")
    print(f"{'='*64}")

    for split in SPLITS:
        texts, labels = _load_split(dataset_name, split, cfg)
        if texts is None:
            print(f"  [{split:10s}]  NOT FOUND")
            continue
        if len(texts) == 0:
            print(f"  [{split:10s}]  EMPTY")
            continue

        n = len(texts)
        avg_words = float(np.mean([len(t.split()) for t in texts]))
        label_counts = {}
        for l in labels:
            label_counts[l] = label_counts.get(l, 0) + 1

        print(f"\n  [{split:10s}]  n={n:>8,}   avg_words={avg_words:.1f}")
        print(f"  {'Class':<10} {'Count':>10} {'%':>8}")
        print(f"  {'-'*32}")
        for cls in sorted(label_counts):
            cnt = label_counts[cls]
            pct = 100.0 * cnt / n
            print(f"  {str(cls):<10} {cnt:>10,} {pct:>7.1f}%")

        row = {
            "dataset": dataset_name,
            "split": split,
            "n": n,
            "avg_words": f"{avg_words:.1f}",
        }
        for cls in sorted(label_counts):
            row[f"class_{cls}_count"] = label_counts[cls]
            row[f"class_{cls}_pct"] = f"{100.0 * label_counts[cls] / n:.1f}"
        rows.append(row)
    return rows


def main():
    with open(ROOT_PATH / "src" / "datasets.json") as f:
        datasets_config = json.load(f)

    all_rows = []
    for dataset_name, cfg in datasets_config.items():
        all_rows.extend(dataset_stats(dataset_name, cfg))

    if not all_rows:
        print("\nNo data found.")
        return

    # Build a consistent set of columns across all datasets
    priority = ["dataset", "split", "n", "avg_words"]
    extra_cols = sorted({k for row in all_rows for k in row if k not in priority})
    fieldnames = priority + extra_cols

    out_path = ROOT_PATH / "results" / "dataset_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n\nStats saved → {out_path}")


if __name__ == "__main__":
    main()
