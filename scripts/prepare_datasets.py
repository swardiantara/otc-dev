"""
Download and prepare the four benchmark datasets for ordinal text classification.

Datasets downloaded from Hugging Face:
  - amazon_reviews  -> amazon_reviews_multi (English)
  - snli            -> snli
  - sst5            -> SetFit/sst5
  - yelp            -> yelp_review_full

Each dataset is saved as three CSV files:
  data/{name}/{name}_train.csv
  data/{name}/{name}_validation.csv
  data/{name}/{name}_test.csv

src/datasets.json is updated automatically with the correct data path.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT_PATH = Path(__file__).parent.parent
DATA_DIR = ROOT_PATH / "data"
DATASETS_JSON = ROOT_PATH / "src" / "datasets.json"


# ---------------------------------------------------------------------------
# Individual dataset preparers
# ---------------------------------------------------------------------------

def prepare_amazon_reviews(data_dir: Path, overwrite: bool) -> bool:
    out = data_dir / "amazon_reviews"
    if _already_done(out, "amazon_reviews") and not overwrite:
        print("  amazon_reviews: already prepared, skipping (use --overwrite to redo).")
        return False

    print("  Downloading amazon_reviews_multi (en) ...")
    ds = load_dataset("amazon_reviews_multi", "en", trust_remote_code=True)

    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": "train", "validation": "validation", "test": "test"}
    for split_name, hf_split in splits.items():
        df = pd.DataFrame({
            "text": ds[hf_split]["review_body"],
            # stars are 1-5; convert to 0-indexed labels
            "label": [s - 1 for s in ds[hf_split]["stars"]],
        })
        path = out / f"amazon_reviews_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"    Saved {len(df):,} rows -> {path.name}")
    return True


def prepare_snli(data_dir: Path, overwrite: bool) -> bool:
    out = data_dir / "snli"
    if _already_done(out, "snli") and not overwrite:
        print("  snli: already prepared, skipping.")
        return False

    print("  Downloading snli ...")
    ds = load_dataset("snli", trust_remote_code=True)

    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": "train", "validation": "validation", "test": "test"}
    for split_name, hf_split in splits.items():
        df = pd.DataFrame({
            "premise": ds[hf_split]["premise"],
            "hypothesis": ds[hf_split]["hypothesis"],
            "label": ds[hf_split]["label"],
        })
        # Remove examples without a gold label (label == -1)
        df = df[df["label"] != -1].reset_index(drop=True)
        path = out / f"snli_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"    Saved {len(df):,} rows -> {path.name}")
    return True


def prepare_sst5(data_dir: Path, overwrite: bool) -> bool:
    out = data_dir / "sst5"
    if _already_done(out, "sst5") and not overwrite:
        print("  sst5: already prepared, skipping.")
        return False

    print("  Downloading SetFit/sst5 ...")
    ds = load_dataset("SetFit/sst5", trust_remote_code=True)

    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": "train", "validation": "validation", "test": "test"}
    for split_name, hf_split in splits.items():
        df = pd.DataFrame({
            # datasets.json expects column name "sentence"
            "sentence": ds[hf_split]["text"],
            "label": ds[hf_split]["label"],
        })
        path = out / f"sst5_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"    Saved {len(df):,} rows -> {path.name}")
    return True


def prepare_yelp(data_dir: Path, overwrite: bool, val_fraction: float = 0.1) -> bool:
    out = data_dir / "yelp"
    if _already_done(out, "yelp") and not overwrite:
        print("  yelp: already prepared, skipping.")
        return False

    print("  Downloading yelp_review_full ...")
    ds = load_dataset("yelp_review_full", trust_remote_code=True)

    out.mkdir(parents=True, exist_ok=True)

    # yelp_review_full has no validation split; carve one out of train
    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    n_total = len(train_texts)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val

    train_df = pd.DataFrame({
        "text": train_texts[:n_train],
        "label": train_labels[:n_train],
    })
    val_df = pd.DataFrame({
        "text": train_texts[n_train:],
        "label": train_labels[n_train:],
    })
    test_df = pd.DataFrame({
        "text": ds["test"]["text"],
        "label": ds["test"]["label"],
    })

    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        path = out / f"yelp_{split_name}.csv"
        df.to_csv(path, index=False)
        print(f"    Saved {len(df):,} rows -> {path.name}")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _already_done(out_dir: Path, name: str) -> bool:
    return all((out_dir / f"{name}_{split}.csv").exists()
               for split in ["train", "validation", "test"])


def update_datasets_json(data_dir: Path) -> None:
    with open(DATASETS_JSON, "r") as f:
        config = json.load(f)

    data_dir_str = str(data_dir.resolve())
    for dataset_name in config:
        config[dataset_name]["path"] = data_dir_str

    with open(DATASETS_JSON, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n  Updated {DATASETS_JSON.relative_to(ROOT_PATH)} with path: {data_dir_str}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

PREPARERS = {
    # "amazon_reviews" excluded: HuggingFace dataset 'amazon_reviews_multi' is defunct
    # (DefunctDatasetError). The prepare_amazon_reviews() function above is kept so it
    # can be re-enabled once an alternative source is found.
    # "amazon_reviews": prepare_amazon_reviews,
    "snli": prepare_snli,
    "sst5": prepare_sst5,
    "yelp": prepare_yelp,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare ordinal classification datasets from Hugging Face."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=list(PREPARERS.keys()),
        choices=list(PREPARERS.keys()),
        help="Which datasets to prepare (default: all four).",
    )
    parser.add_argument(
        "--data_dir", type=Path, default=DATA_DIR,
        help=f"Root directory for saved CSVs (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download and overwrite existing CSV files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    any_changed = False
    for name in args.datasets:
        print(f"\n[{name}]")
        changed = PREPARERS[name](args.data_dir, args.overwrite)
        any_changed = any_changed or changed

    # Always update datasets.json so paths are correct even on first run
    update_datasets_json(args.data_dir)
    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
