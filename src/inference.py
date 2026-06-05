import argparse
from src.model_coral import CoralModel
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List
import numpy as np
from tqdm import tqdm
import csv
import torch
from datasets import load_dataset
from scipy.stats import kendalltau
from scipy.special import softmax, expit
from collections import Counter
import json
import sys
sys.path.append(os.getcwd())


ROOT_PATH = Path(__file__).parent.parent


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
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], max_length=max_len)


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
        help="Datasets to evaluate (default: all four).",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f"Using device: {device}")

    with open(ROOT_PATH / "src" / "datasets.json", "r") as f:
        datasets = json.load(f)

    output_path_metrics = ROOT_PATH / "src" / "outputs_training" / "output_metrics" / "metrics_test_set.csv"
    output_path_metrics.parent.mkdir(parents=True, exist_ok=True)

    model_checkpoint = "google/bert_uncased_L-2_H-128_A-2"
    model_dir = "google/"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    losses = ["CE", "OLL15", "OLL1", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]

    for dataset_file in args.datasets:
        num_classes = datasets[dataset_file]["num_classes"]
        n_distances = datasets[dataset_file]["n_distances"]
        max_len = datasets[dataset_file]["tok_len"]
        sentence1_key, sentence2_key = datasets[dataset_file]["task"]
        dist_matrix = datasets[dataset_file]["dist"]
        directory_path = datasets[dataset_file]["path"]

        data_path = f"{directory_path}/{dataset_file}"
        dataset = load_dataset(
            'csv',
            data_files={'test': f"{data_path}/{dataset_file}_test.csv"},
        )

        models_path = ROOT_PATH / "src" / "outputs_training" / "output_models" / dataset_file / "saved_models" / model_dir
        if not models_path.is_dir():
            print(f"No saved models found for {dataset_file}, skipping.")
            continue

        saved_models = np.sort(os.listdir(models_path))
        dictpath = {}
        offset = 28

        for path in saved_models:
            loss_name = None
            loss_len = 0
            for loss in losses:
                if loss in path:
                    loss_len = len(loss)
                    loss_name = loss
                    break
            if loss_name is None:
                continue

            n = len(dataset_file) + loss_len + offset
            corrected_path = path[:n] + path[n + 2:] + path[n:n + 2]
            dictpath[corrected_path] = {"path": path, "loss": loss_name}

        if output_path_metrics.is_file():
            dt = pd.read_csv(output_path_metrics, header=None, sep='\n')
            dt = dt[0].str.split(',', expand=True)
        else:
            dt = pd.DataFrame(columns=range(5))

        for path in np.sort(list(dictpath.keys())):
            trained_model = dictpath[path]["path"]
            loss_func = dictpath[path]["loss"]

            if trained_model in list(dt.iloc[:, 3]) if len(dt.columns) > 3 else []:
                continue

            pre_trained_model = f"{model_dir}bert-tiny"

            if loss_func == "CORAL":
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
                if loss_func == "CORAL":
                    predictions_test.extend(
                        (np.column_stack((np.zeros((preds.logits.cpu().detach().numpy().shape[0], 1)),
                                          expit(preds.logits.cpu().detach().numpy()))) > 0.5).sum(axis=1)
                    )
                else:
                    distributions.extend(softmax(preds.logits.cpu().detach().numpy(), axis=1).tolist())
                    predictions_test.extend(preds.logits.argmax(dim=1).tolist())

            dico_logs_ = {}
            evaluate_model(labels=encoded_dataset["test"]["label"], preds=predictions_test)

            new_row = (
                [dataset_file, loss_func, pre_trained_model, trained_model]
                + [dico_logs_[k] for k in dico_logs_ if k != "labels-confusion_matrix"]
            )

            with open(output_path_metrics, "a+", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(new_row)
