"""Shared utilities for extracting [CLS] representations and plotting t-SNE.

Used by both training (validation-set t-SNE) and inference (test-set t-SNE,
plus an OB1-correct subset). Kept dependency-light (torch + sklearn + matplotlib)
so it can be imported from training.py / inference.py without pulling in the
sentence-transformers stack used by finetune_embedding.py.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def _get_cmap(n_classes):
    """Return a discrete 'turbo' colormap, robust across matplotlib versions.

    turbo spans blue->red so adjacent ordinal classes stay visually distinct
    while the global low->high label order is encoded by hue progression.
    """
    try:
        return plt.get_cmap("turbo", n_classes)
    except Exception:  # very new matplotlib removed pyplot.get_cmap's lut arg
        from matplotlib import colormaps
        return colormaps["turbo"].resampled(n_classes)


def _pad_batch(seqs, pad_value=0):
    """Right-pad a list of variable-length int sequences to a rectangular batch.

    Inference tokenization is unpadded (ragged), so batches must be padded before
    tensorizing. Padding ``input_ids`` with 0 is harmless because the matching
    ``attention_mask`` is 0 there, so padded positions are ignored by attention.
    Training tokenization is already max-length padded, making this a no-op.
    """
    max_len = max(len(s) for s in seqs)
    return [list(s) + [pad_value] * (max_len - len(s)) for s in seqs]


@torch.no_grad()
def extract_cls_embeddings(model, encoded_split, device, batch_size=256):
    """Return an (N, H) array of last-layer [CLS] embeddings for a tokenized split.

    ``encoded_split`` is a HuggingFace dataset split with at least ``input_ids``
    and ``attention_mask`` columns (``token_type_ids`` is used when present).
    The [CLS] token of the last hidden layer is used — the same representation the
    auxiliary InfoNCE loss operates on. Ragged (unpadded) batches are padded
    on the fly, so both training (padded) and inference (unpadded) splits work.
    """
    was_training = model.training
    model.eval()
    embeddings = []
    n = len(encoded_split)
    for i in range(0, n, batch_size):
        batch = encoded_split[i:i + batch_size]
        input_ids = torch.tensor(_pad_batch(batch["input_ids"], 0)).to(device)
        kwargs = {
            "attention_mask": torch.tensor(_pad_batch(batch["attention_mask"], 0)).to(device),
            "output_hidden_states": True,
        }
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = torch.tensor(
                _pad_batch(batch["token_type_ids"], 0)).to(device)
        outputs = model(input_ids, **kwargs)
        cls = outputs.hidden_states[-1][:, 0]  # (B, H)
        embeddings.append(cls.float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(embeddings, axis=0)


def plot_tsne(embeddings, labels, title, save_path,
              max_samples=5000, metric="cosine", seed=42):
    """Project embeddings to 2-D with t-SNE and save a scatter plot colored by label.

    Args:
        embeddings: (N, H) array.
        labels:     (N,) ordinal integer labels.
        title:      plot title.
        save_path:  output file (parent dirs are created).
        max_samples: stratified per-class subsample cap; 0/None means use all
                     points (t-SNE is ~O(N^2), so large sets can be very slow).
        metric:     distance metric for t-SNE ('cosine' by default).
    """
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(np.unique(labels))
    n_classes = len(unique_labels)

    # Stratified subsample for tractable t-SNE on large splits.
    if max_samples and len(embeddings) > max_samples:
        n_per_class = max(1, max_samples // max(n_classes, 1))
        rng = np.random.RandomState(seed)
        keep = []
        for c in unique_labels:
            idx = np.where(labels == c)[0]
            keep.append(rng.choice(idx, min(n_per_class, len(idx)), replace=False))
        keep = np.concatenate(keep)
        embeddings, labels = embeddings[keep], labels[keep]
        logger.info("t-SNE subsampled to %d points (%d/class) for %s",
                    len(embeddings), n_per_class, save_path.name)

    if len(embeddings) < 3:
        logger.warning("Too few points (%d) for t-SNE at %s; skipping.",
                       len(embeddings), save_path)
        return

    perplexity = min(30, max(5, len(embeddings) // 100))
    embs_2d = TSNE(
        n_components=2, perplexity=perplexity, max_iter=1000,
        metric=metric, init="pca", random_state=seed,
    ).fit_transform(embeddings)

    cmap = _get_cmap(n_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            embs_2d[mask, 0], embs_2d[mask, 1],
            color=cmap(i), label=f"Class {lbl}", alpha=0.5, s=8, linewidths=0,
        )
    ax.legend(title="Label", loc="best", markerscale=2.5, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved t-SNE plot to %s", save_path)
