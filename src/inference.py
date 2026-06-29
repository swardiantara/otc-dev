import argparse
from src.model_coral import CoralModel
from src.tsne_utils import extract_cls_embeddings, plot_tsne
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List
import numpy as np
from tqdm import tqdm
import csv
import re
import torch
from datasets import load_dataset
from scipy.stats import kendalltau
from scipy.special import softmax, expit
from collections import Counter
import json
import sys
sys.path.append(os.getcwd())


ROOT_PATH = Path(__file__).parent.parent


def get_inference_cols(n_distances: int, num_classes: int) -> list:
    """Return the ordered column list for metrics_test_set.csv for a given dataset."""
    cols = ["dataset", "loss", "pretrained_model", "trained_model",
            "accuracy", "precision", "recall", "f1_score"]
    for k in range(1, n_distances):
        cols.append(f"distance_{k}")
    for k in range(1, n_distances - 1):
        cols.append(f"off-by-{k}-accuracy")
    cols.extend(["mae", "mse", "kendalltau"])
    for k in range(num_classes):
        cols.append(f"distrib-{k}")
    return cols


# Maps predefined column names to the keys used inside dico_logs_.
_INFERENCE_COL_TO_KEY = {
    "accuracy": "labels-accuracy",
    "precision": "labels-precision",
    "recall": "labels-recall",
    "f1_score": "labels-f1_score",
}


def build_inference_row(prefix: list, dico_logs_: dict, inference_cols: list) -> list:
    """Build a CSV row aligned to inference_cols; prefix covers the first 4 identifier columns."""
    prefix_cols = {"dataset", "loss", "pretrained_model", "trained_model"}
    prefix_iter = iter(prefix)
    row = []
    for col in inference_cols:
        if col in prefix_cols:
            row.append(next(prefix_iter))
        else:
            row.append(dico_logs_.get(_INFERENCE_COL_TO_KEY.get(col, col), ""))
    return row


def evaluate_model(labels: List[int], preds: List[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc_labels = accuracy_score(labels, preds)
    cm_labels = confusion_matrix(y_true=labels, y_pred=preds)

    distances = [dist_matrix[preds[k]][labels[k]] for k in range(len(preds))]
    cnt = Counter(distances)

    dico_logs_["labels-accuracy"] = round(acc_labels, 4)
    dico_logs_["labels-precision"] = round(precision, 4)
    dico_logs_["labels-recall"] = round(recall, 4)
    dico_logs_["labels-f1_score"] = round(f1, 4)
    dico_logs_["labels-confusion_matrix"] = cm_labels

    for k in range(1, n_distances):
        dico_logs_[f"distance_{k}"] = round(cnt[k] / len(preds), 4)

    acc = acc_labels
    for k in range(1, n_distances - 1):
        acc += cnt[k] / len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    distances_np = np.array(distances)
    dico_logs_["mae"] = distances_np.sum() / len(distances_np)
    dico_logs_["mse"] = (distances_np ** 2).sum() / len(distances_np)
    dico_logs_["kendalltau"], _ = kendalltau(labels, preds)

    repartitions = Counter(preds)
    for k in range(len(repartitions)):
        dico_logs_[f"distrib-{k}"] = repartitions[k]

    return dico_logs_


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


def preprocess_function_unbatched(examples):
    # Explicit str() cast guards against Arrow/None values that would crash the
    # fast tokenizer with "TextInputSequence must be str".
    if sentence2_key is None:
        return tokenizer(str(examples[sentence1_key]), max_length=max_len)
    return tokenizer(str(examples[sentence1_key]), str(examples[sentence2_key]), max_length=max_len)


def get_distributions(distributions: list, labels: list, correct: bool):
    final_distributions = []
    for dist, lab in zip(distributions, labels):
        if int(np.argmax(dist)) == lab and correct:
            final_distributions.append(dist)
        if int(np.argmax(dist)) != lab and not correct:
            final_distributions.append(dist)
    return final_distributions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained ordinal classification models on test sets."
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["amazon_reviews", "snli", "sst5", "yelp"],
        choices=["amazon_reviews", "snli", "sst5", "yelp"],
        help="Datasets to evaluate (default: all available).",
    )
    parser.add_argument(
        "--tsne_max_samples", type=int, default=5000,
        help="Stratified per-class cap on points used for the test-set t-SNE "
             "plots (default: 5000). Set 0 to plot all samples.",
    )
    parser.add_argument(
        "--no_tsne", action="store_true",
        help="Disable the per-model test-set t-SNE visualizations.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run inference even for models whose _SUCCESS marker already exists.",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    with open(ROOT_PATH / "src" / "datasets.json", "r") as f:
        datasets = json.load(f)

    model_checkpoint = "google/bert_uncased_L-2_H-128_A-2"
    model_dir = "google/"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    losses = ["CE", "OLL15", "OLL1", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL", "BCE", "CORN"]

    for dataset_file in args.datasets:
        num_classes = datasets[dataset_file]["num_classes"]
        n_distances = datasets[dataset_file]["n_distances"]
        inference_cols = get_inference_cols(n_distances, num_classes)

        output_path_metrics = (
            ROOT_PATH / "src" / "outputs_training" / "output_metrics"
            / dataset_file / "metrics_test_set.csv"
        )
        output_path_metrics.parent.mkdir(parents=True, exist_ok=True)

        # Per-model inference artifacts mirror the training layout (sibling of
        # output_models / output_metrics). Each run gets its own folder gated by
        # a _SUCCESS marker that the skip logic checks.
        inference_root = (
            ROOT_PATH / "src" / "outputs_training" / "output_inference" / dataset_file
        )

        max_len = datasets[dataset_file]["tok_len"]
        sentence1_key, sentence2_key = datasets[dataset_file]["task"]
        dist_matrix = datasets[dataset_file]["dist"]
        directory_path = datasets[dataset_file]["path"]

        data_path = f"{directory_path}/{dataset_file}"
        dataset = load_dataset(
            'csv',
            data_files={'test': f"{data_path}/{dataset_file}_test.csv"},
        )

        # Drop rows where any text column is None/null/blank (Arrow type inference
        # can re-introduce nulls even after prepare_datasets.py has cleaned the CSV).
        _text_cols = [c for c in [sentence1_key, sentence2_key] if c is not None]

        def _has_valid_text(example):
            return all(
                example.get(col) is not None
                and str(example.get(col, "")).strip() not in ("", "nan", "none")
                for col in _text_cols
            )

        dataset = dataset.filter(_has_valid_text)

        models_path = ROOT_PATH / "src" / "outputs_training" / "output_models" / dataset_file / "saved_models" / model_dir
        if not models_path.is_dir():
            print(f"No saved models found for {dataset_file}, skipping.")
            continue

        saved_models = np.sort(os.listdir(models_path))
        dictpath = {}

        for path in saved_models:
            # Match the base loss and (optionally) an ordinal-triplet suffix:
            #   ...-OLL2-1_...            -> base "OLL2",  label "OLL2"
            #   ...-OLL2-TRIPa1p5-1_...   -> base "OLL2",  label "OLL2-TRIPa1p5"
            # losses is ordered so OLL15 is tested before OLL1 (substring safety).
            base_loss = None
            loss_name = None
            for loss in losses:
                m = re.search(rf"-{re.escape(loss)}(-TRIP\w*)?-\d+_", path)
                if m:
                    base_loss = loss
                    loss_name = loss + (m.group(1) or "")
                    break
            if loss_name is None:
                continue

            dictpath[path] = {"path": path, "loss": loss_name, "base_loss": base_loss}

        if output_path_metrics.is_file():
            dt = pd.read_csv(output_path_metrics, header=0)
        else:
            dt = pd.DataFrame(columns=inference_cols)

        for path in np.sort(list(dictpath.keys())):
            trained_model = dictpath[path]["path"]
            loss_func = dictpath[path]["loss"]        # full label, e.g. "OLL2-TRIPa1p5"
            base_loss = dictpath[path]["base_loss"]   # base loss, e.g. "OLL2" (drives the head)

            # File-based skip: a run is "done" once its _SUCCESS marker exists.
            artifact_dir = inference_root / trained_model
            success_marker = artifact_dir / "_SUCCESS"
            if success_marker.is_file() and not args.force:
                print(f"Skipping {dataset_file}/{trained_model} (already inferred)")
                continue

            pre_trained_model = f"{model_dir}bert-tiny"

            if base_loss == "CORAL":
                model = CoralModel.from_pretrained(
                    str(models_path / trained_model)).to(device)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(models_path / trained_model)).to(device)

            encoded_dataset = dataset.map(preprocess_function_unbatched, batched=False)

            batch_size = 1
            predictions_test, distributions = [], []

            for k in tqdm(range(0, len(encoded_dataset["test"]), batch_size), desc="Predicting"):
                inputs = {
                    "input_ids": encoded_dataset["test"][k:k + batch_size]["input_ids"],
                    "attention_mask": encoded_dataset["test"][k:k + batch_size]["attention_mask"],
                    "token_type_ids": encoded_dataset["test"][k:k + batch_size]["token_type_ids"],
                }
                with torch.no_grad():
                    preds = model(
                        torch.tensor(inputs["input_ids"]).to(device),
                        attention_mask=torch.tensor(inputs["attention_mask"]).to(device),
                        token_type_ids=torch.tensor(inputs["token_type_ids"]).to(device),
                    )
                if base_loss == "CORAL":
                    coral_logits = preds.logits.cpu().detach().numpy()
                    # P(y > k) per cumulative threshold; stored as raw probabilities.
                    distributions.extend(expit(coral_logits).tolist())
                    predictions_test.extend(
                        (np.column_stack((np.zeros((coral_logits.shape[0], 1)),
                                          expit(coral_logits))) > 0.5).sum(axis=1)
                    )
                elif base_loss == "CORN":
                    # CORN: rank-consistent cumulative probs P(y>k)=prod sigmoid(logit_j);
                    # class = number of thresholds exceeding 0.5.
                    cum_probs = np.cumprod(expit(preds.logits.cpu().detach().numpy()), axis=1)
                    distributions.extend(cum_probs.tolist())
                    predictions_test.extend((cum_probs > 0.5).sum(axis=1).tolist())
                elif base_loss == "BCE":
                    # BCE ordinal: sigmoid per position, class = (positions > 0.5) - 1
                    # target[k]=1 iff k<=y, so predicted y = count(sigmoid>0.5) - 1
                    sigmoid_out = expit(preds.logits.cpu().detach().numpy())
                    distributions.extend(sigmoid_out.tolist())
                    raw_preds = (sigmoid_out > 0.5).sum(axis=1) - 1
                    predictions_test.extend(np.clip(raw_preds, 0, num_classes - 1).tolist())
                else:
                    distributions.extend(softmax(preds.logits.cpu().detach().numpy(), axis=1).tolist())
                    predictions_test.extend(preds.logits.argmax(dim=1).tolist())

            dico_logs_ = {}
            evaluate_model(labels=encoded_dataset["test"]["label"], preds=predictions_test)

            # ---- Shared metrics_test_set.csv (append once per model) ----
            already_in_csv = (
                "trained_model" in dt.columns
                and trained_model in dt["trained_model"].tolist()
            )
            if not already_in_csv:
                prefix = [dataset_file, loss_func, pre_trained_model, trained_model]
                new_row = build_inference_row(prefix, dico_logs_, inference_cols)
                write_header = (
                    not output_path_metrics.is_file()
                    or output_path_metrics.stat().st_size == 0
                )
                with open(output_path_metrics, "a+", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(inference_cols)
                    writer.writerow(new_row)

            # ---- Per-model inference artifacts (mirrors training layout) ----
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # metrics.json: all scalar metrics (confusion matrix saved separately).
            metrics_out = {"dataset": dataset_file, "loss": loss_func,
                           "trained_model": trained_model}
            for key, val in dico_logs_.items():
                if key == "labels-confusion_matrix":
                    continue
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                elif isinstance(val, np.floating):
                    val = float(val)
                elif isinstance(val, np.integer):
                    val = int(val)
                metrics_out[key] = val
            with open(artifact_dir / "metrics.json", "w") as f:
                json.dump(metrics_out, f, indent=2)

            # Raw per-sample prediction probabilities + confusion matrix.
            np.save(artifact_dir / "probabilities.npy", np.array(distributions))
            cm = dico_logs_.get("labels-confusion_matrix")
            if cm is not None:
                np.savetxt(artifact_dir / "confusion_matrix.csv",
                           np.asarray(cm), fmt="%d", delimiter=",")

            # Test-set t-SNE: all samples, plus the OB1-correct subset
            # (ordinal distance between prediction and true label <= 1).
            if not args.no_tsne:
                try:
                    labels_arr = np.array(encoded_dataset["test"]["label"])
                    preds_arr = np.array(predictions_test)
                    test_embs = extract_cls_embeddings(
                        model, encoded_dataset["test"], device, batch_size=256)
                    plot_tsne(
                        test_embs, labels_arr, title=f"{trained_model} (test)",
                        save_path=artifact_dir / "tsne_test.pdf",
                        max_samples=args.tsne_max_samples)
                    ob1_mask = np.array([
                        dist_matrix[int(p)][int(t)] <= 1
                        for p, t in zip(preds_arr, labels_arr)
                    ])
                    if ob1_mask.any():
                        plot_tsne(
                            test_embs[ob1_mask], labels_arr[ob1_mask],
                            title=f"{trained_model} (test, OB1-correct)",
                            save_path=artifact_dir / "tsne_test_ob1.pdf",
                            max_samples=args.tsne_max_samples)
                except Exception as e:
                    print(f"[t-SNE] test plot failed for {trained_model}: {e}")

            # Mark the run fully processed — gates the skip on future runs.
            success_marker.write_text("ok\n")
