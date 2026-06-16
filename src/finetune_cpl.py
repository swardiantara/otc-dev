"""Constrained Proxies Learning (CPL) for ordinal *text* classification.

This is a faithful port of the AAAI-2023 paper

    "Controlling Class Layout for Deep Ordinal Classification via
     Constrained Proxies Learning"  (Wang et al., 2023)
    code: https://github.com/Tenvence/cpl

adapted from the original image setting (VGG-16 feature extractor on
Adience-Face / Historical-Color / Image-Aesthetics) to the text setting used
throughout this repository (BERT-tiny feature extractor on SST-5 / SNLI /
Amazon / Yelp).

Idea
----
Instead of fine-tuning a contrastive embedding with hand-built pairs (as in
``finetune_embedding.py``), CPL learns one *proxy* vector per ordinal class and
explicitly **constrains the global layout of those proxies** so that the class
order is reflected as a geometric order in feature space.  A sample is assigned
to the class whose proxy is most similar to its feature, which — when the
proxies lie on an ordinal layout — yields the unimodal probability distribution
that is ideal for ordinal classification.

Two families of layout constraints are implemented (exactly as in the paper):

  * Hard-CPL  – the proxy generator is *structurally* forced onto an ordinal
                layout:
                    H-L : linear layout      p_k = k * v0      (Euclidean metric)
                    H-S : semicircular layout (Eq. 8)          (cosine metric)
                trained with the basic KL loss  L_basic = KL( Q(k*) || P(f) ).

  * Soft-CPL  – proxies are learned freely (p_k = v_k) but the proxy-to-proxies
                similarity distribution is regularised to be unimodal:
                    S-P : Poisson  smoothing
                    S-B : Binomial smoothing
                trained with  L = CE(P(f), k*) + alpha * KL( U(k*) || Q(k*) ).

The components (proxies learners, metrics, losses) are ported verbatim from the
reference repository; only the feature extractor and the data / evaluation
plumbing are new (text-specific, and matching this repo's conventions).

Outputs
-------
Test-set metrics are appended to

    src/outputs_training/output_metrics/{dataset}/metrics_test_set.csv

in the *same* schema produced by ``src/inference.py`` (so that
``scripts/analyze_results.py`` can aggregate CPL rows alongside the other
losses).  The per-run identifier is encoded in the ``trained_model`` column as

    {alias}-{dataset}-{loss_tag}-{seed}_{epochs}_ep_{lr}_lr_{batch}_batch

which ``analyze_results.extract_run_info`` parses for seed / lr / epochs.
"""

import argparse
import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import kendalltau
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent


# ==========================================================================
# 1. Metric methods  (ported from cpl/metric_methods.py)
# ==========================================================================
class CosineMetric(nn.Module):
    """Scaled pairwise cosine similarity:  s * cos(x1_i, x2_j).  (Eq. 6)"""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x1, x2):
        return self.scale * torch.cosine_similarity(x1[:, None, :], x2[None, :, :], dim=-1)


class EuclideanMetric(nn.Module):
    """Student-t style similarity:  -log(1 + ||x1_i - x2_j||^2).  (Eq. 5)"""

    @staticmethod
    def forward(x1, x2):
        return -torch.log(1 + torch.cdist(x1, x2) ** 2)


# ==========================================================================
# 2. Proxies learners  (ported from cpl/proxies_learner.py)
# ==========================================================================
class BaseProxiesLearner(nn.Module):
    """Free proxies p_k = v_k  (used by Soft-CPL).  (Eq. 9)"""

    def __init__(self, num_ranks, dim):
        super().__init__()
        self.proxies = nn.Parameter(torch.empty((num_ranks, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.proxies)

    def forward(self):
        return self.proxies


class LinearProxiesLearner(nn.Module):
    """Linear layout p_k = k * v0  (Hard-CPL, Euclidean).  (Eq. 7)"""

    def __init__(self, num_ranks, dim):
        super().__init__()
        self.rank_ids = nn.Parameter(
            torch.arange(num_ranks)[:, None].float(), requires_grad=False
        )
        self.v0 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.v0)

    def forward(self):
        return self.rank_ids * self.v0


class SemicircularProxiesLearner(nn.Module):
    """Semicircular layout (Hard-CPL, cosine).  (Eq. 8)"""

    def __init__(self, num_ranks, dim):
        super().__init__()
        self.num_ranks = num_ranks
        self.rank_ids = nn.Parameter(
            torch.arange(num_ranks)[:, None].float(), requires_grad=False
        )
        self.v0 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        self.v1 = nn.Parameter(torch.empty((1, dim)), requires_grad=True)
        nn.init.xavier_normal_(self.v0)
        nn.init.xavier_normal_(self.v1)

    def forward(self):
        theta = self.rank_ids * np.pi / (self.num_ranks - 1)
        gamma = torch.cosine_similarity(self.v0, self.v1).arccos()
        norm_v0 = self.v0 / torch.linalg.norm(self.v0, dim=-1)
        norm_v1 = self.v1 / torch.linalg.norm(self.v1, dim=-1)
        proxies = (gamma - theta).sin() / gamma.sin() * norm_v0 + theta.sin() / gamma.sin() * norm_v1
        return proxies


# ==========================================================================
# 3. Criterions  (ported from cpl/criterions.py)
# ==========================================================================
class SoftCplPoissonLoss(nn.Module):
    """CE(P(f), k*) + lam * KL( U_poisson(k*) || Q(k*) ).  (Eq. 13 + Eq. 15)"""

    def __init__(self, num_ranks, tau, loss_lam):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        self.loss_lam = loss_lam

    def forward(self, assign_metric, gt, proxies_metric):
        loss = F.cross_entropy(assign_metric, gt)

        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        factorial = torch.tensor(scipy.special.factorial(rank_ids))  # [1, C]
        rank_ids = rank_ids.to(gt.device)
        factorial = factorial.to(gt.device)

        lam = torch.arange(self.num_ranks)[:, None].float() + 0.5  # [C, 1]
        lam = lam.to(gt.device)
        ordinal_smoothing_func = rank_ids * torch.log(lam) - lam - torch.log(factorial)
        target_distribution = F.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss += self.loss_lam * F.kl_div(
            F.softmax(proxies_metric, dim=-1).log(),
            target_distribution,
            reduction="batchmean",
        )
        return loss


class SoftCplBinomialLoss(nn.Module):
    """CE(P(f), k*) + lam * KL( U_binomial(k*) || Q(k*) ).  (Eq. 13 + Eq. 17)"""

    def __init__(self, num_ranks, tau, loss_lam):
        super().__init__()
        self.num_ranks = num_ranks
        self.tau = tau
        self.loss_lam = loss_lam

    def forward(self, assign_metric, gt, proxies_metric):
        loss = F.cross_entropy(assign_metric, gt)

        rank_ids = torch.arange(self.num_ranks)[None, :].float()  # [1, C]
        binom = scipy.special.binom(self.num_ranks - 1, rank_ids)  # [1, C]
        p = (2 * torch.arange(self.num_ranks)[:, None].float() + 1) / (2 * self.num_ranks)

        rank_ids = rank_ids.to(gt.device)
        binom = binom.to(gt.device)
        p = p.to(gt.device)
        ordinal_smoothing_func = (
            binom.log()
            + rank_ids * p.log()
            + (self.num_ranks - 1 - rank_ids) * (1 - p).log()
        )
        target_distribution = F.softmax(ordinal_smoothing_func / self.tau, dim=-1)

        loss += self.loss_lam * F.kl_div(
            F.softmax(proxies_metric, dim=-1).log(),
            target_distribution,
            reduction="batchmean",
        )
        return loss


class HardCplLoss(nn.Module):
    """L_basic = KL( Q(k*) || P(f) ), with Q(k*) the (detached) gt proxy row.  (Eq. 3)"""

    @staticmethod
    def forward(assign_metric, gt, proxies_metric):
        selected_proxies_metric = proxies_metric[gt, :].detach()  # [B, C]
        loss = F.kl_div(
            F.log_softmax(assign_metric, dim=-1),
            F.softmax(selected_proxies_metric, dim=-1),
            reduction="batchmean",
        )
        return loss


# ==========================================================================
# 4. Text feature extractor + CPL model
# ==========================================================================
class TextFeatureExtractor(nn.Module):
    """BERT encoder + linear projection to `feature_dim`.

    Plays the role of VGG-16 (encoder) followed by the replaced final fc layer
    in the original paper.  Mean-pooling over the (mask-aware) token embeddings
    is used as the sentence representation, which is robust for tiny BERT.
    """

    def __init__(self, model_name, feature_dim):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(hidden_size, feature_dim)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        token_embeddings = out.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts  # mean pooling
        return self.projection(pooled)  # [B, feature_dim]


class CplModel(nn.Module):
    """Wraps a feature extractor, a proxies learner and a metric method.

    Mirrors cpl/cpl_model.py: the forward pass returns the sample-to-proxies
    similarity (`assign_metric`, used for classification) and the
    proxy-to-proxies similarity (`proxies_metric`, used for layout constraints).
    """

    def __init__(self, feature_extractor, proxies_learner, metric_method):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.proxies_learner = proxies_learner
        self.metric_method = metric_method

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        feature = self.feature_extractor(input_ids, attention_mask, token_type_ids)
        proxies = self.proxies_learner()
        assign_metric = self.metric_method(feature, proxies)
        proxies_metric = self.metric_method(proxies, proxies)
        return assign_metric, proxies_metric


def build_model_and_criterion(num_ranks, args):
    """Map the `--constraint` (and `--metric_method`) flag to concrete
    components, exactly as utils.get_model_criterion does in the reference."""
    feature_extractor = TextFeatureExtractor(args.model_checkpoint, args.feature_dim)

    if args.constraint == "S-P":
        proxies_learner = BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = SoftCplPoissonLoss(num_ranks, args.tau, args.loss_lam)
        metric_method = _soft_metric(args)
    elif args.constraint == "S-B":
        proxies_learner = BaseProxiesLearner(num_ranks, args.feature_dim)
        criterion = SoftCplBinomialLoss(num_ranks, args.tau, args.loss_lam)
        metric_method = _soft_metric(args)
    elif args.constraint == "H-L":
        proxies_learner = LinearProxiesLearner(num_ranks, args.feature_dim)
        criterion = HardCplLoss()
        metric_method = EuclideanMetric()  # linear layout is Euclidean-specific
    elif args.constraint == "H-S":
        proxies_learner = SemicircularProxiesLearner(num_ranks, args.feature_dim)
        criterion = HardCplLoss()
        metric_method = CosineMetric(args.cosine_scale)  # semicircle is cosine-specific
    else:
        raise ValueError(f"Unknown constraint: {args.constraint}")

    model = CplModel(feature_extractor, proxies_learner, metric_method)
    return model, criterion


def _soft_metric(args):
    if args.metric_method == "C":
        return CosineMetric(args.cosine_scale)
    return EuclideanMetric()


# ==========================================================================
# 5. Data
# ==========================================================================
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_split(dataset_name, split, cfg):
    """Load one split's texts + labels, mirroring training.py's cleaning."""
    data_path = Path(cfg["path"]) / dataset_name
    raw = load_dataset(
        "csv",
        data_files={split: str(data_path / f"{dataset_name}_{split}.csv")},
    )[split]

    col1, col2 = cfg["task"]
    texts1 = [str(t) for t in raw[col1]]
    texts2 = [str(t) for t in raw[col2]] if col2 is not None else None
    labels = list(raw["label"])

    # Drop rows with invalid labels (e.g. SNLI label == -1) or blank text.
    bad = {"", "nan", "none"}
    keep1, keep2, keep_lbl = [], [], []
    for i, lbl in enumerate(labels):
        if lbl is None or int(lbl) < 0:
            continue
        if texts1[i].strip().lower() in bad:
            continue
        if texts2 is not None and texts2[i].strip().lower() in bad:
            continue
        keep1.append(texts1[i])
        if texts2 is not None:
            keep2.append(texts2[i])
        keep_lbl.append(int(lbl))

    text_pair = keep2 if texts2 is not None else None
    return keep1, text_pair, keep_lbl


def encode(tokenizer, texts1, texts2, max_len):
    if texts2 is None:
        return tokenizer(texts1, truncation=True, padding="max_length", max_length=max_len)
    return tokenizer(texts1, texts2, truncation=True, padding="max_length", max_length=max_len)


# ==========================================================================
# 6. Train / eval loops  (adapted from engine.py)
# ==========================================================================
def run_epoch(model, criterion, optimizer, scheduler, loader, device, scaler):
    model.train()
    total_loss, n = 0.0, 0
    use_amp = scaler is not None
    for batch in loader:
        labels = batch.pop("label").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            assign_metric, proxies_metric = model(**inputs)
            loss = criterion(assign_metric, labels, proxies_metric)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, gts = [], []
    for batch in loader:
        labels = batch.pop("label")
        inputs = {k: v.to(device) for k, v in batch.items()}
        # assign_metric is the sample-to-proxies similarity; the predicted rank
        # is the proxy most similar to the feature (Eq. 4).
        assign_metric, _ = model(**inputs)
        preds.extend(torch.argmax(assign_metric, dim=-1).cpu().tolist())
        gts.extend(labels.tolist())
    return np.array(preds), np.array(gts)


# ==========================================================================
# 7. Metrics  (same schema as src/inference.py)
# ==========================================================================
def get_inference_cols(n_distances, num_classes):
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


def evaluate(preds, labels, dist_matrix, n_distances, num_classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    distances = [dist_matrix[int(preds[k])][int(labels[k])] for k in range(len(preds))]
    cnt = Counter(distances)

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
    for k in range(1, n_distances):
        metrics[f"distance_{k}"] = round(cnt[k] / len(preds), 4)
    running = acc
    for k in range(1, n_distances - 1):
        running += cnt[k] / len(preds)
        metrics[f"off-by-{k}-accuracy"] = round(running, 4)

    distances_np = np.array(distances, dtype=float)
    metrics["mae"] = distances_np.mean()
    metrics["mse"] = (distances_np ** 2).mean()
    tau, _ = kendalltau(labels, preds)
    metrics["kendalltau"] = tau

    repartitions = Counter(int(p) for p in preds)
    for k in range(num_classes):
        metrics[f"distrib-{k}"] = repartitions.get(k, 0)
    return metrics


# ==========================================================================
# 8. Main
# ==========================================================================
def loss_tag(args):
    """Human-readable identifier for the CPL variant (used in the CSV)."""
    if args.constraint in ("H-L", "H-S"):
        return f"CPL-{args.constraint}"
    return f"CPL-{args.constraint}-{args.metric_method}"  # soft: append metric


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    with open(ROOT_PATH / "src" / "datasets.json") as f:
        datasets_config = json.load(f)
    cfg = datasets_config[args.dataset]
    num_classes = cfg["num_classes"]
    n_distances = cfg["n_distances"]
    dist_matrix = cfg["dist"]
    max_len = args.max_len or cfg["tok_len"]

    # Resolve tau per the paper if left unset (S-P: 0.11, S-B: 0.13).
    if args.tau is None:
        args.tau = 0.13 if args.constraint == "S-B" else 0.11

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)

    logger.info("Loading and tokenizing datasets ...")
    tr_t1, tr_t2, tr_lbl = load_split(args.dataset, "train", cfg)
    va_t1, va_t2, va_lbl = load_split(args.dataset, "validation", cfg)
    te_t1, te_t2, te_lbl = load_split(args.dataset, "test", cfg)

    train_enc = encode(tokenizer, tr_t1, tr_t2, max_len)
    val_enc = encode(tokenizer, va_t1, va_t2, max_len)
    test_enc = encode(tokenizer, te_t1, te_t2, max_len)

    train_ds = TextDataset(train_enc, tr_lbl)
    val_ds = TextDataset(val_enc, va_lbl)
    test_ds = TextDataset(test_enc, te_lbl)
    logger.info(
        f"Dataset '{args.dataset}': {len(train_ds)} train / {len(val_ds)} val / "
        f"{len(test_ds)} test samples, {num_classes} classes."
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    metrics_dir = (
        ROOT_PATH / "src" / "outputs_training" / "output_metrics" / args.dataset
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / "metrics_test_set.csv"
    inference_cols = get_inference_cols(n_distances, num_classes)

    model_alias = args.model_alias or args.model_checkpoint.split("/")[-1]
    tag = loss_tag(args)

    for seed in args.seeds:
        trained_model = (
            f"{model_alias}-{args.dataset}-{tag}-{seed}"
            f"_{args.epochs}_ep_{args.lr}_lr_{args.batch_size}_batch"
        )

        # Skip if this exact run is already recorded.
        if metrics_csv.is_file():
            import pandas as pd
            done = pd.read_csv(metrics_csv)
            if "trained_model" in done.columns and trained_model in done["trained_model"].tolist():
                logger.info(f"Skipping (already done): {trained_model}")
                continue

        logger.info(f"===== {args.dataset} | {tag} | seed={seed} =====")
        set_seed(seed)

        model, criterion = build_model_and_criterion(num_classes, args)
        model.to(device)
        criterion.to(device) if isinstance(criterion, nn.Module) else None

        # Two param groups: the proxies learner (and any metric params) are
        # trained at a higher learning rate than the BERT feature extractor,
        # following the paper (lr_feature << lr_proxies).
        feat_params = [
            p for n, p in model.named_parameters()
            if n.startswith("feature_extractor") and p.requires_grad
        ]
        proxy_params = [
            p for n, p in model.named_parameters()
            if not n.startswith("feature_extractor") and p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": feat_params, "lr": args.lr},
                {"params": proxy_params, "lr": args.lr * args.lr_pl_mul},
            ],
            weight_decay=args.weight_decay,
        )

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )

        scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # Train; keep the weights that minimise validation MAE (paper protocol).
        best_val_mae = float("inf")
        best_state = None
        epochs_no_improve = 0
        for epoch in range(args.epochs):
            train_loss = run_epoch(
                model, criterion, optimizer, scheduler, train_loader, device, scaler
            )
            val_preds, val_gts = predict(model, val_loader, device)
            val_mae = np.mean(np.abs(val_preds - val_gts))
            val_acc = accuracy_score(val_gts, val_preds) * 100
            logger.info(
                f"[epoch {epoch + 1}/{args.epochs}] loss={train_loss:.4f} "
                f"val_acc={val_acc:.2f} val_mae={val_mae:.4f}"
            )
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if args.early_stopping_patience and epochs_no_improve >= args.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (best val MAE={best_val_mae:.4f}).")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final test-set evaluation.
        test_preds, test_gts = predict(model, test_loader, device)
        metrics = evaluate(test_preds, test_gts, dist_matrix, n_distances, num_classes)
        logger.info(
            f"TEST  acc={metrics['accuracy']} mae={metrics['mae']:.4f} "
            f"mse={metrics['mse']:.4f} tau={metrics['kendalltau']}"
        )

        # Append a row in the inference.py schema.
        prefix = {
            "dataset": args.dataset,
            "loss": tag,
            "pretrained_model": args.model_checkpoint,
            "trained_model": trained_model,
        }
        row = [prefix.get(col, metrics.get(col, "")) for col in inference_cols]
        write_header = not metrics_csv.is_file() or metrics_csv.stat().st_size == 0
        with open(metrics_csv, "a+", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(inference_cols)
            writer.writerow(row)
        logger.info(f"Metrics appended to {metrics_csv}")

        # Optionally persist the trained weights.
        if args.save_model:
            save_dir = (
                ROOT_PATH / "src" / "outputs_training" / "output_models"
                / args.dataset / "cpl_saved_models" / trained_model
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "pytorch_model.pt")
            with open(save_dir / "cpl_args.json", "w") as f:
                json.dump({**vars(args), "loss_tag": tag, "seed": seed}, f, indent=2)
            logger.info(f"Model saved to {save_dir}")

    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(description="Constrained Proxies Learning for ordinal text classification")
    parser.add_argument("--dataset", required=True,
                        choices=["sst5", "snli", "amazon_reviews", "yelp"])
    parser.add_argument("--constraint", default="S-B", choices=["S-P", "S-B", "H-L", "H-S"],
                        help="S-P/S-B: Soft-CPL (Poisson/Binomial); H-L/H-S: Hard-CPL (Linear/Semicircular).")
    parser.add_argument("--metric_method", default="E", choices=["E", "C"],
                        help="Similarity for Soft-CPL only: E=Euclidean, C=Cosine "
                             "(ignored for hard constraints, which fix the metric).")
    parser.add_argument("--model_checkpoint", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--model_alias", default=None,
                        help="Short name used in the CSV identifier (default: last path segment).")

    # CPL hyperparameters (paper defaults).
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--cosine_scale", type=float, default=6.0, help="scale s for cosine metric (Eq. 6).")
    parser.add_argument("--tau", type=float, default=None,
                        help="Soft-CPL shape parameter; default 0.11 (S-P) / 0.13 (S-B).")
    parser.add_argument("--loss_lam", type=float, default=6.0, help="alpha tradeoff for the unimodal loss (Eq. 13).")

    # Optimisation.
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of the BERT feature extractor.")
    parser.add_argument("--lr_pl_mul", type=float, default=10.0,
                        help="multiplier giving the proxies-learner learning rate (lr * lr_pl_mul).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=None, help="override tokenizer max length (default: dataset tok_len).")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop if val MAE does not improve for this many epochs (0 disables).")

    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--save_model", action="store_true", help="Persist the best weights per run.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
