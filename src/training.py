import argparse
from scipy.special import expit
import json
from collections import Counter
import random
from datasets import load_dataset
import csv
from tqdm import tqdm
import numpy as np
from src.model_coral import CoralModel
from src.loss_functions import (
    OLL1Trainer, OLL15Trainer, OLL2Trainer,
    WKLTrainer, SOFT2Trainer, SOFT3Trainer, SOFT4Trainer, EMDTrainer,
)
import os
import sys
import torch
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
sys.path.append(os.getcwd())


ROOT_PATH = Path(__file__).parent.parent


def compute_metrics_coral(pred):
    labels = pred.label_ids
    preds = (np.column_stack((np.zeros((pred.predictions.shape[0], 1)), expit(
        pred.predictions))) > 0.5).sum(axis=1)
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

    for k in np.sort(list(cnt.keys()))[1:]:
        dico_logs_[f"distance_{k}"] = round(cnt[k] / len(preds), 4)

    acc = acc_labels
    for k in np.sort(list(cnt.keys()))[1:-1]:
        acc += cnt[k] / len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    return dico_logs_


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
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

    for k in np.sort(list(cnt.keys()))[1:]:
        dico_logs_[f"distance_{k}"] = round(cnt[k] / len(preds), 4)

    acc = acc_labels
    for k in np.sort(list(cnt.keys()))[1:-1]:
        acc += cnt[k] / len(preds)
        dico_logs_[f"off-by-{k}-accuracy"] = round(acc, 4)

    return dico_logs_


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True, padding='max_length', max_length=max_len)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding='max_length', max_length=max_len)


losses_dict = {
    "CE": Trainer,
    "OLL1": OLL1Trainer,
    "OLL15": OLL15Trainer,
    "OLL2": OLL2Trainer,
    "WKL": WKLTrainer,
    "SOFT2": SOFT2Trainer,
    "SOFT3": SOFT3Trainer,
    "SOFT4": SOFT4Trainer,
    "EMD": EMDTrainer,
    "CORAL": Trainer,
}

ALL_DATASETS = ["sst5", "amazon_reviews", "yelp", "snli"]
ALL_LOSSES = ["CE", "OLL1", "OLL15", "OLL2", "WKL", "SOFT2", "SOFT3", "SOFT4", "EMD", "CORAL"]
ALL_LRS = [1e-4, 7.5e-5, 5e-5, 2.5e-5, 1e-5]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ordinal text classification models with various loss functions."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
        help="Datasets to train on (default: all four).",
    )
    parser.add_argument(
        "--losses", nargs="+", default=ALL_LOSSES, choices=ALL_LOSSES,
        help="Loss functions to use (default: all ten).",
    )
    parser.add_argument(
        "--learning_rates", nargs="+", type=float, default=ALL_LRS,
        metavar="LR", help="Learning rates to sweep (default: 5 values from paper).",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        help="Random seeds for repeated runs (default: 1 2 3 4 5).",
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
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    for data_file in args.datasets:
        for loss_type in args.losses:
            for learning_rate_ in args.learning_rates:
                num_classes = datasets[data_file]["num_classes"]
                max_len = datasets[data_file]["tok_len"]
                sentence1_key, sentence2_key = datasets[data_file]["task"]
                dist_matrix = datasets[data_file]["dist"]
                directory_path = datasets[data_file]["path"]

                data_path = f"{directory_path}/{data_file}"
                dataset = load_dataset(
                    'csv',
                    data_files={
                        'train': f"{data_path}/{data_file}_train.csv",
                        'validation': f"{data_path}/{data_file}_validation.csv",
                        'test': f"{data_path}/{data_file}_test.csv",
                    },
                )

                encoded_dataset = dataset.map(preprocess_function, batched=True)

                dico_logs_ = {}

                weight_decay_ = 0.01
                train_batch_size_ = 1024
                valid_batch_size_ = 1024

                epochs_ = max(1, int(20_000_000 / len(dataset['train'])))
                stopping_rate = max(1, int(0.05 * epochs_))

                for k in tqdm(args.seeds, desc=f"{data_file}/{loss_type}/lr={learning_rate_}"):
                    random.seed(k)
                    np.random.seed(k)
                    torch.manual_seed(k)

                    model_name = "-".join([
                        model_checkpoint, data_file, loss_type, str(k)
                    ])
                    save_dir = (
                        ROOT_PATH / "src" / "outputs_training" / "output_models"
                        / data_file / "saved_models"
                        / f"{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch"
                    )

                    if save_dir.is_dir():
                        print(f"Skipping {save_dir.name} (already exists)")
                        continue

                    if loss_type == "CORAL":
                        model = CoralModel.from_pretrained(
                            model_checkpoint, num_labels=num_classes).to(device)
                    else:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_checkpoint, num_labels=num_classes).to(device)
                    model.dist_matrix = dist_matrix

                    print(f'Epochs: {epochs_} | Learning rate: {learning_rate_}')
                    dico_logs_["model_name"] = model_name
                    dico_logs_["epochs"] = epochs_
                    dico_logs_["learning_rate"] = learning_rate_
                    dico_logs_["train_batch_size"] = train_batch_size_

                    output_dir = (
                        ROOT_PATH / "src" / "outputs_training" / "output_models"
                        / data_file / "training" / model_name
                    )
                    log_dir = (
                        ROOT_PATH / "src" / "outputs_training" / "output_models"
                        / data_file / "logs"
                        / f"{model_name}_{epochs_}_ep_{learning_rate_}_lr_{train_batch_size_}_batch"
                    )

                    training_args = TrainingArguments(
                        output_dir=str(output_dir),
                        fp16=use_cuda,
                        num_train_epochs=epochs_,
                        evaluation_strategy="epoch",
                        save_strategy="epoch",
                        load_best_model_at_end=True,
                        logging_steps=50,
                        save_total_limit=1,
                        per_device_train_batch_size=train_batch_size_,
                        per_device_eval_batch_size=valid_batch_size_,
                        learning_rate=learning_rate_,
                        weight_decay=weight_decay_,
                        logging_dir=str(log_dir),
                    )

                    loss_function = losses_dict[loss_type]
                    eval_function = compute_metrics_coral if loss_type == "CORAL" else compute_metrics

                    trainer = loss_function(
                        model=model,
                        args=training_args,
                        compute_metrics=eval_function,
                        train_dataset=encoded_dataset["train"],
                        eval_dataset=encoded_dataset["validation"],
                        callbacks=[EarlyStoppingCallback(early_stopping_patience=stopping_rate)],
                    )

                    print('--- TRAINING ---')
                    trainer.train()
                    print('--- EVALUATION ---')
                    trainer.evaluate()

                    trainer.save_model(str(save_dir))

                    output_path_metrics = (
                        ROOT_PATH / "src" / "outputs_training" / "output_metrics"
                        / "fine_tuning_metrics.csv"
                    )
                    output_path_metrics.parent.mkdir(parents=True, exist_ok=True)
                    values = [dico_logs_[key] for key in dico_logs_ if key != "labels-confusion_matrix"]
                    with open(output_path_metrics, "a+", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(values)
