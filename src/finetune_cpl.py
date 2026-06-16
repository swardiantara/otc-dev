"""Constrained Proxies Learning (CPL) — *embedding* fine-tuning for ordinal text.

A faithful port of the AAAI-2023 paper

    "Controlling Class Layout for Deep Ordinal Classification via
     Constrained Proxies Learning"  (Wang et al., 2023)
    code: https://github.com/Tenvence/cpl

adapted to the **embedding fine-tuning stage** of this repository, mirroring
``src/finetune_embedding.py``:

  * CPL is used purely to *shape the BERT-tiny embedding space* so that the
    ordinal class order becomes a geometric order.  One learnable proxy per
    class is constrained onto an ordinal layout, and the (mean-pooled) BERT
    embedding is pulled towards the proxy of its class.
  * The fine-tuned embedding is **not** turned into a classifier here.  Instead
    it is evaluated with NOS and kNN-MAE on the validation set (the same
    metrics as ``finetune_embedding.py``) and the BERT backbone is pushed to the
    HuggingFace Hub for later use by the classifier pipeline (``src/training.py``
    via ``scripts/run_finetuning.sh``).

Design choices (matching the surrounding repo conventions):
  * The embedding that is shaped, evaluated and pushed is the **mean-pooled BERT
    output** (dim = hidden size = 128). No extra projection: the proxies live in
    that space, which is exactly what kNN-MAE measures and what the downstream
    classifier reuses.
  * The proxies learner is trained at a higher learning rate than the BERT
    encoder (``--lr × --lr_pl_mul``), as in the paper.

CPL variants (``--constraint``):
    H-L : Hard-CPL, linear layout       p_k = k·v0           (Euclidean metric)
    H-S : Hard-CPL, semicircular layout (Eq. 8)              (cosine metric)
    S-P : Soft-CPL, Poisson  smoothing                       (--metric_method E/C)
    S-B : Soft-CPL, Binomial smoothing                       (--metric_method E/C)

Outputs
-------
  results/cpl_embedding/{dataset}/{model_id}.csv     – NOS + kNN-MAE on validation
  results/cpl_embedding/{dataset}/{model_id}_tsne.pdf
  HuggingFace Hub: {user}/{model_id}                 – fine-tuned BERT backbone

where  model_id = {alias}-{dataset}-cpl-{constraint}-{metric}
(e.g. ``bert-tiny-sst5-cpl-S-B-E``); ``scripts/recap_cpl.py`` parses these
filenames to pick the best embedding per dataset by kNN-MAE.
"""

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
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
# 4. Feature extractor + CPL model
# ==========================================================================
class TextFeatureExtractor(nn.Module):
    """BERT encoder with mask-aware mean pooling.

    The embedding is the mean-pooled last-hidden-state (dim = hidden size, 128
    for BERT-tiny).  No projection: this is the embedding that CPL shapes, that
    kNN-MAE measures, and that is pushed to the Hub for the classifier stage.
    """

    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        token_embeddings = out.last_hidden_state  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts  # mean pooling -> [B, H]


class CplModel(nn.Module):
    """Feature extractor + proxies learner + metric (cf. cpl/cpl_model.py).

    Forward returns the sample-to-proxies similarity (`assign_metric`) and the
    proxy-to-proxies similarity (`proxies_metric`) used by the layout losses.
    """

    def __init__(self, feature_extractor, proxies_learner, metric_method):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.proxies_learner = proxies_learner
        self.metric_method = metric_method

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        return self.feature_extractor(input_ids, attention_mask, token_type_ids)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        feature = self.encode(input_ids, attention_mask, token_type_ids)
        proxies = self.proxies_learner()
        assign_metric = self.metric_method(feature, proxies)
        proxies_metric = self.metric_method(proxies, proxies)
        return assign_metric, proxies_metric


def resolve_metric_tag(args):
    """The similarity metric actually used: fixed for hard constraints, chosen
    by --metric_method for soft constraints."""
    if args.constraint == "H-L":
        return "E"
    if args.constraint == "H-S":
        return "C"
    return args.metric_method


def build_model_and_criterion(num_ranks, args):
    """Map --constraint to concrete components (cf. utils.get_model_criterion).

    The proxies live in the BERT mean-pooled embedding space, so the proxy
    dimension equals the encoder hidden size (128 for BERT-tiny).
    """
    feature_extractor = TextFeatureExtractor(args.model_name)
    dim = feature_extractor.embedding_dim

    if args.constraint == "S-P":
        proxies_learner = BaseProxiesLearner(num_ranks, dim)
        criterion = SoftCplPoissonLoss(num_ranks, args.tau, args.loss_lam)
        metric_method = _soft_metric(args)
    elif args.constraint == "S-B":
        proxies_learner = BaseProxiesLearner(num_ranks, dim)
        criterion = SoftCplBinomialLoss(num_ranks, args.tau, args.loss_lam)
        metric_method = _soft_metric(args)
    elif args.constraint == "H-L":
        proxies_learner = LinearProxiesLearner(num_ranks, dim)
        criterion = HardCplLoss()
        metric_method = EuclideanMetric()  # linear layout is Euclidean-specific
    elif args.constraint == "H-S":
        proxies_learner = SemicircularProxiesLearner(num_ranks, dim)
        criterion = HardCplLoss()
        metric_method = CosineMetric(args.cosine_scale)  # semicircle is cosine-specific
    else:
        raise ValueError(f"Unknown constraint: {args.constraint}")

    model = CplModel(feature_extractor, proxies_learner, metric_method)
    return model, criterion


def _soft_metric(args):
    return CosineMetric(args.cosine_scale) if args.metric_method == "C" else EuclideanMetric()


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
    """Load one split's texts (+ optional second text column) and labels."""
    data_path = Path(cfg["path"]) / dataset_name
    raw = load_dataset(
        "csv",
        data_files={split: str(data_path / f"{dataset_name}_{split}.csv")},
    )[split]

    col1, col2 = cfg["task"]
    texts1 = [str(t) for t in raw[col1]]
    texts2 = [str(t) for t in raw[col2]] if col2 is not None else None
    labels = list(raw["label"])

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

    return keep1, (keep2 if texts2 is not None else None), keep_lbl


def encode(tokenizer, texts1, texts2, max_len):
    if texts2 is None:
        return tokenizer(texts1, truncation=True, padding="max_length", max_length=max_len)
    return tokenizer(texts1, texts2, truncation=True, padding="max_length", max_length=max_len)


# ==========================================================================
# 6. Embedding evaluation: NOS + kNN-MAE  (same metrics as finetune_embedding)
# ==========================================================================
def compute_nos(embeddings, labels, k_values=(1, 3, 5, 10), metric="cosine"):
    """Neighborhood Ordinal Smoothness: mean |label - neighbor label| over kNN."""
    labels = np.asarray(labels)
    max_k = max(k_values)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    scores = {}
    for k in k_values:
        neighbor_labels = labels[indices[:, 1:k + 1]]
        scores[f"NOS_{k}"] = float(np.mean(np.abs(labels[:, None] - neighbor_labels)))
    scores["NOS_avg"] = float(np.mean([scores[f"NOS_{k}"] for k in k_values]))
    return scores


def compute_knn_mae(embeddings, labels, k_values=(1, 3, 5, 10), metric="cosine"):
    """Parameter-free ordinal classifier: predict the (rounded) median label of
    the k nearest validation neighbours; report MAE against the true label."""
    labels = np.asarray(labels)
    max_k = max(k_values)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    scores = {}
    for k in k_values:
        predicted = np.round(np.median(labels[indices[:, 1:k + 1]], axis=1)).astype(int)
        scores[f"kNN_MAE_{k}"] = float(np.mean(np.abs(predicted - labels)))
    scores["kNN_MAE_avg"] = float(np.mean([scores[f"kNN_MAE_{k}"] for k in k_values]))
    return scores


def plot_tsne(embeddings, labels, model_id, save_path, metric="cosine", max_samples=5000):
    """t-SNE scatter of validation embeddings, coloured by ordinal label."""
    labels = np.asarray(labels)
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)

    if len(embeddings) > max_samples:
        n_per_class = max_samples // n_classes
        rng = np.random.RandomState(42)
        keep = np.concatenate([
            rng.choice(np.where(labels == c)[0],
                       min(n_per_class, int(np.sum(labels == c))), replace=False)
            for c in unique_labels
        ])
        embeddings, labels = embeddings[keep], labels[keep]

    perplexity = min(30, max(5, len(embeddings) // 100))
    embs_2d = TSNE(
        n_components=2, perplexity=perplexity, max_iter=1000,
        metric=metric, init="pca", random_state=42,
    ).fit_transform(embeddings)

    try:  # matplotlib >= 3.9 removed cm.get_cmap
        cmap = plt.get_cmap("turbo").resampled(n_classes)
    except AttributeError:
        cmap = plt.cm.get_cmap("turbo", n_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], color=cmap(i),
                   label=f"Class {lbl}", alpha=0.5, s=8, linewidths=0)
    ax.legend(title="Label", loc="best", markerscale=2.5, fontsize=9)
    ax.set_title(f"t-SNE  |  {model_id}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"t-SNE plot saved to {save_path}")


@torch.no_grad()
def encode_embeddings(model, loader, device):
    """Return mean-pooled BERT embeddings for every sample in the loader."""
    model.eval()
    embs = []
    for batch in loader:
        batch.pop("label", None)
        inputs = {k: v.to(device) for k, v in batch.items()}
        embs.append(model.encode(**inputs).float().cpu().numpy())
    return np.concatenate(embs, axis=0)


# ==========================================================================
# 7. Training loop  (standalone, two learning rates)
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


# ==========================================================================
# 8. Main
# ==========================================================================
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
    max_len = args.max_len or cfg["tok_len"]

    if args.tau is None:  # paper defaults
        args.tau = 0.13 if args.constraint == "S-B" else 0.11

    # NOS / kNN-MAE neighbour metric: cosine when the CPL metric is cosine,
    # euclidean otherwise (keeps evaluation consistent with the learned space).
    metric_tag = resolve_metric_tag(args)
    nn_metric = "cosine" if metric_tag == "C" else "euclidean"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading and tokenizing datasets ...")
    tr_t1, tr_t2, tr_lbl = load_split(args.dataset, "train", cfg)
    va_t1, va_t2, va_lbl = load_split(args.dataset, "validation", cfg)

    train_ds = TextDataset(encode(tokenizer, tr_t1, tr_t2, max_len), tr_lbl)
    val_ds = TextDataset(encode(tokenizer, va_t1, va_t2, max_len), va_lbl)
    val_labels = np.array(va_lbl)
    logger.info(
        f"Dataset '{args.dataset}': {len(train_ds)} train / {len(val_ds)} val samples, "
        f"{num_classes} classes."
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model, criterion = build_model_and_criterion(num_classes, args)
    model.to(device)
    if isinstance(criterion, nn.Module):
        criterion.to(device)

    # Two param groups: proxies learner (and any metric params) get the higher
    # learning rate; the BERT encoder is fine-tuned gently.
    feat_params = [p for n, p in model.named_parameters()
                   if n.startswith("feature_extractor") and p.requires_grad]
    proxy_params = [p for n, p in model.named_parameters()
                    if not n.startswith("feature_extractor") and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": feat_params, "lr": args.lr},
            {"params": proxy_params, "lr": args.lr * args.lr_pl_mul},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    model_alias = args.model_alias or args.model_name.split("/")[-1]
    model_id = f"{model_alias}-{args.dataset}-cpl-{args.constraint}-{metric_tag}"

    metrics_dir = ROOT_PATH / "results" / "cpl_embedding" / args.dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Train; keep the weights that minimise validation kNN-MAE (the embedding
    # selection metric used to pick the best CPL variant downstream).
    best_knn_mae = float("inf")
    best_state = None
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, criterion, optimizer, scheduler, train_loader, device, scaler)
        val_embs = encode_embeddings(model, val_loader, device)
        nos = compute_nos(val_embs, val_labels, metric=nn_metric)
        knn = compute_knn_mae(val_embs, val_labels, metric=nn_metric)
        logger.info(
            f"[epoch {epoch + 1}/{args.epochs}] loss={train_loss:.4f} "
            f"val_NOS_avg={nos['NOS_avg']:.4f} val_kNN_MAE_avg={knn['kNN_MAE_avg']:.4f}"
        )
        if knn["kNN_MAE_avg"] < best_knn_mae:
            best_knn_mae = knn["kNN_MAE_avg"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if args.early_stopping_patience and epochs_no_improve >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (best kNN-MAE={best_knn_mae:.4f}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final validation metrics on the best embedding.
    val_embs = encode_embeddings(model, val_loader, device)
    final_nos = compute_nos(val_embs, val_labels, metric=nn_metric)
    final_knn = compute_knn_mae(val_embs, val_labels, metric=nn_metric)
    all_metrics = {**final_nos, **final_knn}
    logger.info(f"Final validation metrics: {all_metrics}")

    metrics_csv = metrics_dir / f"{model_id}.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics.keys()))
        writer.writeheader()
        writer.writerow(all_metrics)
    logger.info(f"Metrics saved to {metrics_csv}")

    tsne_pdf = metrics_dir / f"{model_id}_tsne.pdf"
    plot_tsne(val_embs, val_labels, model_id, tsne_pdf, metric=nn_metric)

    # Persist locally if requested.
    if args.save_model:
        save_dir = ROOT_PATH / "src" / "outputs_training" / "cpl_embeddings" / model_id
        save_dir.mkdir(parents=True, exist_ok=True)
        model.feature_extractor.encoder.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info(f"Backbone saved locally to {save_dir}")

    # Push the fine-tuned BERT backbone to the Hub (so src/training.py can load
    # it via AutoModelForSequenceClassification.from_pretrained). Mirrors
    # finetune_embedding.py: resolve the user, recreate the repo, upload weights
    # + tokenizer + the validation metrics and t-SNE plot.
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            hf_username = api.whoami()["name"]
        except Exception as e:
            raise RuntimeError("Not logged into HuggingFace. Run `huggingface-cli login` first.") from e
        model_hub_id = f"{hf_username}/{model_id}"

        logger.info(f"Pushing backbone to Hub: {model_hub_id}")
        if api.repo_exists(repo_id=model_hub_id, repo_type="model"):
            api.delete_repo(repo_id=model_hub_id, repo_type="model")
        model.feature_extractor.encoder.push_to_hub(model_hub_id)
        tokenizer.push_to_hub(model_hub_id)
        api.upload_file(
            path_or_fileobj=str(metrics_csv),
            path_in_repo="cpl_embedding_metrics.csv",
            repo_id=model_hub_id, repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=str(tsne_pdf),
            path_in_repo="tsne_validation.pdf",
            repo_id=model_hub_id, repo_type="model",
        )
        logger.info("Push complete.")

    logger.info("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Constrained Proxies Learning — embedding fine-tuning for ordinal text classification"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["sst5", "snli", "amazon_reviews", "yelp"])
    parser.add_argument("--constraint", default="S-B", choices=["S-P", "S-B", "H-L", "H-S"],
                        help="S-P/S-B: Soft-CPL (Poisson/Binomial); H-L/H-S: Hard-CPL (Linear/Semicircular).")
    parser.add_argument("--metric_method", default="E", choices=["E", "C"],
                        help="Similarity for Soft-CPL only: E=Euclidean, C=Cosine "
                             "(ignored for hard constraints, which fix the metric).")
    parser.add_argument("--model_name", default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--model_alias", default=None,
                        help="Short name used in model_id / Hub repo (default: last path segment).")

    # CPL hyperparameters (paper defaults).
    parser.add_argument("--cosine_scale", type=float, default=6.0, help="scale s for cosine metric (Eq. 6).")
    parser.add_argument("--tau", type=float, default=None,
                        help="Soft-CPL shape parameter; default 0.11 (S-P) / 0.13 (S-B).")
    parser.add_argument("--loss_lam", type=float, default=6.0, help="alpha tradeoff for the unimodal loss (Eq. 13).")

    # Optimisation.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of the BERT feature extractor.")
    parser.add_argument("--lr_pl_mul", type=float, default=10.0,
                        help="multiplier giving the proxies-learner learning rate (lr * lr_pl_mul).")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=None, help="override tokenizer max length (default: dataset tok_len).")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                        help="Stop if val kNN-MAE does not improve for this many epochs (0 disables).")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true", help="Also save the backbone + tokenizer locally.")
    parser.add_argument("--push_to_hub", dest="push_to_hub", action="store_true", default=True,
                        help="Push the fine-tuned backbone to the HuggingFace Hub (default: on).")
    parser.add_argument("--no_push", dest="push_to_hub", action="store_false",
                        help="Disable pushing to the Hub (e.g. for local testing).")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
