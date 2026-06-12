import argparse
import csv
import json
import logging
import math
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
# from sentence_transformers import SentenceTransformer, models
from sentence_transformers.sentence_transformer.modules import SentenceTransformer, models
# from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
# from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parent.parent


# ==========================================
# 1. NOS (Neighborhood Ordinal Smoothness) — shared computation
# ==========================================
def compute_nos(embeddings, labels, k_values=[1, 3, 5, 10], metric='cosine'):
    labels = np.array(labels)
    max_k = max(k_values)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    scores = {}
    for k in k_values:
        neighbor_labels = labels[indices[:, 1:k+1]]
        scores[f"NOS_{k}"] = float(np.mean(np.abs(labels[:, None] - neighbor_labels)))
    scores["NOS_avg"] = float(np.mean([scores[f"NOS_{k}"] for k in k_values]))
    return scores


class NOSEvaluator(SentenceEvaluator):
    def __init__(self, sentences, labels, k_values=[1, 3, 5, 10], metric='cosine', name="val"):
        self.sentences = sentences
        self.labels = np.array(labels)
        self.k_values = k_values
        self.metric = metric
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        logger.info(f"Running NOS Evaluation (epoch={epoch})")
        embeddings = model.encode(self.sentences, convert_to_numpy=True, show_progress_bar=False)
        nos_scores = compute_nos(embeddings, self.labels, self.k_values, self.metric)
        logger.info(f"NOS: {nos_scores}")
        # All scores are logged; NOS_avg is the checkpointing criterion via metric_for_best_model
        return {f"{self.name}_{key}": val for key, val in nos_scores.items()}


# ==========================================
# 2. Custom Loss: Ordinal Proxy Contrastive (Hadsell 2006 with ordinal-aware margin)
# ==========================================
class OrdinalProxyContrastiveLoss(nn.Module):
    def __init__(self, model, margin_type='adaptive', distance_metric='cosine',
                 max_margin=1.0, fixed_margin=0.5):
        super().__init__()
        self.model = model

        if distance_metric == 'euclidean':
            self._dist = lambda e1, e2: F.pairwise_distance(e1, e2)
        else:  # cosine
            self._dist = lambda e1, e2: 1 - F.cosine_similarity(e1, e2)

        if margin_type == 'adaptive':
            self._margin = lambda nd: nd * max_margin
        else:
            self._margin = lambda nd: torch.full_like(nd, fixed_margin)

    def forward(self, sentence_features, labels):
        emb1 = self.model(sentence_features[0])['sentence_embedding']
        emb2 = self.model(sentence_features[1])['sentence_embedding']
        is_positive = labels[:, 0].float()
        normalized_label_dist = labels[:, 1].float()
        distances = self._dist(emb1, emb2)
        margin = self._margin(normalized_label_dist)
        loss_pos = is_positive * distances.pow(2)
        loss_neg = (1 - is_positive) * torch.clamp(margin - distances, min=0.0).pow(2)
        return (loss_pos + loss_neg).mean()


# ==========================================
# 3. Data Loading
# ==========================================
def load_split(dataset_name, split, datasets_config):
    cfg = datasets_config[dataset_name]
    data_path = Path(cfg["path"]) / dataset_name
    raw = load_dataset(
        'csv',
        data_files={split: str(data_path / f"{dataset_name}_{split}.csv")},
    )[split]

    text_col1, text_col2 = cfg["task"]
    if text_col2 is not None:
        # Two-sentence datasets (e.g., SNLI): concatenate with separator
        texts = [f"{p} [SEP] {h}" for p, h in zip(raw[text_col1], raw[text_col2])]
    else:
        texts = list(raw[text_col1])

    labels = list(raw["label"])
    # Filter rows with invalid labels (e.g., SNLI label=-1 for no consensus)
    valid = [(t, l) for t, l in zip(texts, labels) if l >= 0]
    texts, labels = zip(*valid) if valid else ([], [])
    return list(texts), list(labels)


# ==========================================
# 4. Proxy Selection & Pair Construction
# ==========================================
def _select_proxies(embeddings, class_indices, k_proxies, metric='cosine'):
    """Return k_proxies medoid indices (into the original embeddings array)."""
    c_emb = embeddings[class_indices]
    if k_proxies == 1:
        centroid = c_emb.mean(axis=0, keepdims=True)
        dists = pairwise_distances(c_emb, centroid, metric=metric)
        return [class_indices[int(np.argmin(dists))]]
    kmeans = KMeans(n_clusters=k_proxies, random_state=42, n_init=10).fit(c_emb)
    proxies = []
    for center in kmeans.cluster_centers_:
        dists = pairwise_distances(c_emb, center.reshape(1, -1), metric=metric)
        proxies.append(class_indices[int(np.argmin(dists))])
    return proxies


def build_pairs(texts, labels, embeddings, k_proxies, max_label_diff, metric='cosine'):
    labels_arr = np.array(labels)
    texts = list(texts)

    class_proxies = {
        int(c): _select_proxies(embeddings, np.where(labels_arr == c)[0], k_proxies, metric)
        for c in np.unique(labels_arr)
    }

    pairs = {"text_a": [], "text_b": [], "label": []}

    for i in range(len(texts)):
        lbl = int(labels_arr[i])
        own_proxies = class_proxies[lbl]

        # Positive pair: sample → closest proxy of its own class
        if k_proxies == 1:
            proxy_idx = own_proxies[0]
        else:
            dists = pairwise_distances(embeddings[i:i+1], embeddings[own_proxies], metric=metric)
            proxy_idx = own_proxies[int(np.argmin(dists))]
        pairs["text_a"].append(texts[i])
        pairs["text_b"].append(texts[proxy_idx])
        pairs["label"].append([1.0, 0.0])

        # Negative pairs: sample → proxy of each other class (sample-proxy, not sample-sample)
        for other_c, other_proxies in class_proxies.items():
            if other_c == lbl:
                continue
            norm_dist = abs(lbl - other_c) / max_label_diff
            for p_idx in other_proxies:
                pairs["text_a"].append(texts[i])
                pairs["text_b"].append(texts[p_idx])
                pairs["label"].append([0.0, norm_dist])

    # For k>1: pull sub-cluster proxies of the same class together
    if k_proxies > 1:
        for proxies in class_proxies.values():
            for m1 in proxies:
                for m2 in proxies:
                    if m1 != m2:
                        pairs["text_a"].append(texts[m1])
                        pairs["text_b"].append(texts[m2])
                        pairs["label"].append([1.0, 0.0])

    return Dataset.from_dict(pairs)


# ==========================================
# 5. kNN-MAE (parameter-free ordinal classifier evaluation)
# ==========================================
def compute_knn_mae(embeddings, labels, k_values=[1, 3, 5, 10], metric='cosine'):
    labels = np.array(labels)
    max_k = max(k_values)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric=metric).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    scores = {}
    for k in k_values:
        # Median keeps the prediction within the discrete label space (unlike mean)
        predicted = np.round(np.median(labels[indices[:, 1:k+1]], axis=1)).astype(int)
        scores[f"kNN_MAE_{k}"] = float(np.mean(np.abs(predicted - labels)))
    scores["kNN_MAE_avg"] = float(np.mean([scores[f"kNN_MAE_{k}"] for k in k_values]))
    return scores


# ==========================================
# 6. t-SNE validation-set visualization
# ==========================================
def plot_tsne(embeddings, labels, model_id, save_path, metric='cosine', max_samples=5000):
    """Project validation embeddings to 2-D with t-SNE and save a scatter plot.

    Large datasets are stratified-subsampled to max_samples before running t-SNE
    so that runtime stays reasonable regardless of validation-set size.
    The distance metric matches the one used during fine-tuning.
    """
    labels = np.array(labels)
    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)

    # Stratified subsample: keep up to max_samples // n_classes points per class
    if len(embeddings) > max_samples:
        n_per_class = max_samples // n_classes
        rng = np.random.RandomState(42)
        keep = []
        for c in unique_labels:
            idx = np.where(labels == c)[0]
            keep.append(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
        keep = np.concatenate(keep)
        embeddings, labels = embeddings[keep], labels[keep]
        logger.info(f"t-SNE subsampled to {len(embeddings)} points ({n_per_class} per class)")

    perplexity = min(30, max(5, len(embeddings) // 100))
    logger.info(f"Running t-SNE (n={len(embeddings)}, perplexity={perplexity}, metric={metric})...")
    embs_2d = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        metric=metric,
        init='pca',
        random_state=42,
    ).fit_transform(embeddings)

    # Sequential colormap so ordinal distance is visually encoded
    cmap = plt.cm.get_cmap('viridis', n_classes)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1],
            color=cmap(i),
            label=f"Class {lbl}",
            alpha=0.5,
            s=8,
            linewidths=0,
        )

    ax.legend(title="Label", loc='best', markerscale=2.5, fontsize=9)
    ax.set_title(f"t-SNE  |  {model_id}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"t-SNE plot saved to {save_path}")


# ==========================================
# 7. Cosine → Euclidean margin conversion
# ==========================================
def cosine_to_euclidean_margin(cosine_margin: float) -> float:
    """Convert a margin expressed in cosine-distance space to the equivalent
    euclidean-distance margin for unit-normalized embeddings.

    Derivation (unit vectors a, b):
        d_euc² = ||a - b||² = 2 - 2·cos(θ) = 2·d_cos
        d_euc  = sqrt(2·d_cos)

    Examples (cosine → euclidean):
        max_margin   1.0  → sqrt(2) ≈ 1.4142
        fixed_margin 0.5  → sqrt(1) = 1.0
    """
    return math.sqrt(2.0 * cosine_margin)


# ==========================================
# 7. Main Training Pipeline
# ==========================================
def main(args):
    with open(ROOT_PATH / "src" / "datasets.json") as f:
        datasets_config = json.load(f)

    if args.dataset not in datasets_config:
        raise ValueError(f"Dataset '{args.dataset}' not in datasets.json")

    cfg = datasets_config[args.dataset]
    max_label_diff = cfg["num_classes"] - 1

    # Margins are always specified in cosine-distance space; convert when euclidean is used.
    if args.distance_metric == 'euclidean':
        args.max_margin = cosine_to_euclidean_margin(args.max_margin)
        args.fixed_margin = cosine_to_euclidean_margin(args.fixed_margin)
        logger.info(
            f"Euclidean margins (converted from cosine): "
            f"max_margin={args.max_margin:.4f}, fixed_margin={args.fixed_margin:.4f}"
        )

    logger.info("Loading datasets from local CSVs...")
    train_texts, train_labels = load_split(args.dataset, "train", datasets_config)
    val_texts, val_labels = load_split(args.dataset, "validation", datasets_config)

    logger.info("Initializing model...")
    word_emb = models.Transformer(args.model_name, max_seq_length=128)
    pooling = models.Pooling(word_emb.get_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_emb, pooling])

    logger.info("Computing initial embeddings for proxy selection...")
    train_embeddings = model.encode(
        train_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=512
    )

    logger.info(f"Building training pairs (k_proxies={args.k_proxies})...")
    train_dataset = build_pairs(
        train_texts, train_labels, train_embeddings, args.k_proxies, max_label_diff,
        metric=args.distance_metric,
    )

    k_values = [1, 3, 5, 10]
    evaluator = NOSEvaluator(
        list(val_texts), list(val_labels),
        k_values=k_values, metric=args.distance_metric, name="val",
    )
    loss = OrdinalProxyContrastiveLoss(
        model=model,
        margin_type=args.margin_type,
        distance_metric=args.distance_metric,
        max_margin=args.max_margin,
        fixed_margin=args.fixed_margin,
    )

    # model_alias flows into the HuggingFace hub model ID and into training.py's
    # model_checkpoint argument, so it appears correctly in fine_tuning_metrics.csv
    # for ablation tracking — no local weight storage is needed.
    model_alias = args.model_alias or args.model_name.split('/')[-1]
    model_id = f"{model_alias}-{args.dataset}-k{args.k_proxies}-{args.margin_type}-{args.distance_metric}"

    # Resolve HuggingFace username and generate hub model ID from hyperparameters
    api = HfApi()
    try:
        hf_username = api.whoami()["name"]
    except Exception as e:
        raise RuntimeError(
            "Not logged into HuggingFace. Run `huggingface-cli login` first."
        ) from e

    model_hub_id = f"{hf_username}/{model_id}"

    # Checkpoints are written to a temp dir and discarded after training;
    # the best weights live in HuggingFace, not on local disk.
    with tempfile.TemporaryDirectory(prefix="contrastive_ckpt_") as tmp_ckpt_dir:
        training_args = SentenceTransformerTrainingArguments(
            output_dir=tmp_ckpt_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_val_NOS_avg",
            greater_is_better=False,
            push_to_hub=False,
        )

        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            evaluator=evaluator,
            loss=loss,
        )

        logger.info("Starting contrastive fine-tuning...")
        trainer.train()
    # Temp checkpoints are deleted here; model holds the best weights in memory.

    # Final evaluation on best model: NOS + kNN-MAE on validation set
    logger.info("Computing final NOS and kNN-MAE on validation set...")
    val_embs = model.encode(
        list(val_texts), convert_to_numpy=True, show_progress_bar=True, batch_size=512
    )
    val_labels_arr = np.array(list(val_labels))

    final_nos = compute_nos(val_embs, val_labels_arr, k_values, metric=args.distance_metric)
    knn_mae = compute_knn_mae(val_embs, val_labels_arr, k_values, metric=args.distance_metric)

    all_metrics = {**final_nos, **knn_mae}
    logger.info(f"Final validation metrics: {all_metrics}")

    # Save validation metrics to results/embedding/{dataset}/{model_id}.csv
    # The filename encodes all hyperparameters; used to pick the best embedding
    # configuration before the classifier training stage.
    metrics_dir = ROOT_PATH / "results" / "embedding" / args.dataset
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = metrics_dir / f"{model_id}.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics.keys()))
        writer.writeheader()
        writer.writerow(all_metrics)
    logger.info(f"Metrics saved to {metrics_csv}")

    tsne_png = metrics_dir / f"{model_id}_tsne.pdf"
    plot_tsne(val_embs, val_labels_arr, model_id, tsne_png, metric=args.distance_metric)

    # Push model weights and metrics CSV to HuggingFace.
    # exist_ok=True handles both first-time repo creation and re-pushes cleanly,
    # without triggering a duplicate-repo error from an earlier create_repo call.
    logger.info(f"Pushing model to Hub: {model_hub_id}")
    model.push_to_hub(model_hub_id, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(metrics_csv),
        path_in_repo="contrastive_metrics.csv",
        repo_id=model_hub_id,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=str(tsne_png),
        path_in_repo="tsne_validation.pdf",
        repo_id=model_hub_id,
        repo_type="model",
    )
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ordinal Proxy Contrastive Fine-Tuning")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["sst5", "snli", "amazon_reviews", "yelp"],
    )
    parser.add_argument("--model_name", type=str, default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--model_alias", type=str, default=None,
                        help="Short model name used in hub ID and CSV logs (e.g. bert-tiny). "
                             "Defaults to the last segment of --model_name.")
    parser.add_argument("--k_proxies", type=int, default=1,
                        help="Number of proxy medoids per class")
    parser.add_argument("--margin_type", type=str, choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--distance_metric", type=str, choices=["euclidean", "cosine"], default="cosine")
    parser.add_argument("--max_margin", type=float, default=1.0,
                        help="Max margin for adaptive setting")
    parser.add_argument("--fixed_margin", type=float, default=0.5,
                        help="Margin for fixed setting")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)

    args = parser.parse_args()
    main(args)
