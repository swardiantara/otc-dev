import torch.nn.functional as F
import numpy as np
from transformers import Trainer
import torch
import sys
import os
sys.path.append(os.getcwd())


def _unwrap(model):
    """Return the underlying model regardless of DataParallel/DDP wrapping."""
    return model.module if hasattr(model, 'module') else model


class OLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=device, requires_grad=True)
        err = -torch.log(1 - probas) * abs(distances_tensor) ** 2
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class nOLL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[dist_matrix[true_labels[j][i]][label_ids[j][i]] / np.sum([dist_matrix[n][label_ids[j][i]] for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=device, requires_grad=True)
        err = -torch.log(1 - probas) * abs(distances_tensor) ** 2
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class WKLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        y_pred = F.softmax(logits, dim=1)
        label_vec = torch.arange(0, num_classes, dtype=torch.float, device=device)
        row_label_vec = torch.reshape(label_vec, (1, num_classes))
        col_label_vec = torch.reshape(label_vec, (num_classes, 1))
        col_mat = torch.tile(col_label_vec, (1, num_classes))
        row_mat = torch.tile(row_label_vec, (num_classes, 1))
        weight_mat = (col_mat - row_mat) ** 2
        y_true = F.one_hot(labels, num_classes=num_classes).float()
        batch_size = y_true.shape[0]
        cat_labels = torch.matmul(y_true, col_label_vec)
        cat_label_mat = torch.tile(cat_labels, [1, num_classes])
        row_label_mat = torch.tile(row_label_vec, [batch_size, 1])
        weight = (cat_label_mat - row_label_mat) ** 2
        numerator = torch.sum(weight * y_pred)
        label_dist = torch.sum(y_true, axis=0)
        pred_dist = torch.sum(y_pred, axis=0)
        w_pred_dist = torch.t(torch.matmul(weight_mat, pred_dist))
        denominator = torch.sum(torch.matmul(label_dist, w_pred_dist / batch_size), axis=0)
        loss = torch.log(numerator / denominator + 1e-7)
        return (loss, outputs) if return_outputs else loss


class SOFT10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-10*dist_matrix[true_labels[j][i]][label_ids[j][i]]) / np.sum([np.exp(-10*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs, device=device, requires_grad=True)
        err = -torch.log(probas) * softs_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class SOFT5Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-5*dist_matrix[true_labels[j][i]][label_ids[j][i]]) / np.sum([np.exp(-5*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs, device=device, requires_grad=True)
        err = -torch.log(probas) * softs_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class SOFT2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-2*dist_matrix[true_labels[j][i]][label_ids[j][i]]) / np.sum([np.exp(-2*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs, device=device, requires_grad=True)
        err = -torch.log(probas) * softs_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class SOFT3Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-3*dist_matrix[true_labels[j][i]][label_ids[j][i]]) / np.sum([np.exp(-3*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs, device=device, requires_grad=True)
        err = -torch.log(probas) * softs_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class SOFT4Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        softs = [[np.exp(-4*dist_matrix[true_labels[j][i]][label_ids[j][i]]) / np.sum([np.exp(-4*dist_matrix[n][label_ids[j][i]]) for n in range(num_classes)]) for i in range(num_classes)] for j in range(len(labels))]
        softs_tensor = torch.tensor(softs, device=device, requires_grad=True)
        err = -torch.log(probas) * softs_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class OLL1Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=device, requires_grad=True)
        err = -torch.log(1 - probas) * distances_tensor
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class OLL15Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        dist_matrix = m.dist_matrix
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        true_labels = [num_classes*[labels[k].item()] for k in range(len(labels))]
        label_ids = len(labels)*[[k for k in range(num_classes)]]
        distances = [[float(dist_matrix[true_labels[j][i]][label_ids[j][i]]) for i in range(num_classes)] for j in range(len(labels))]
        distances_tensor = torch.tensor(distances, device=device, requires_grad=True)
        err = -torch.log(1 - probas) * distances_tensor ** 1.5
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


class BCEOrdinalTrainer(Trainer):
    """Ordinal BCE loss.

    Each sample y is encoded as a K-dimensional cumulative binary vector:
      target[k] = 1  if k <= y  (i.e. the first y+1 positions are 1, the rest 0)
    Example for K=5: label 0 -> [1,0,0,0,0], label 4 -> [1,1,1,1,1].

    A sigmoid is applied to each of the K logits independently and BCE is
    computed element-wise.  At inference time the predicted class is the
    number of positions that exceed 0.5, minus 1 (clamped to [0, K-1]).
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, K)

        # Build cumulative binary targets via broadcasting: target[i,k] = (k <= y_i)
        k = torch.arange(num_classes, device=device)          # (K,)
        targets = (k <= labels.unsqueeze(1)).float()           # (batch, K)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return (loss, outputs) if return_outputs else loss


class EMDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        m = _unwrap(model)
        num_classes = m.num_labels
        labels = inputs["labels"]
        device = labels.device
        outputs = model(**inputs)
        logits = outputs.logits
        probas = F.softmax(logits, dim=1)
        CDF_pred = torch.cumsum(probas, dim=1)
        CDF_true = torch.tensor(
            [labels[k].item() * [0.] + (num_classes - labels[k].item()) * [1.] for k in range(len(labels))],
            device=device, requires_grad=True
        )
        err = (CDF_pred - CDF_true) ** 2
        loss = torch.sum(err, axis=1).mean()
        return (loss, outputs) if return_outputs else loss


# ===========================================================================
# Ordinal-aware auxiliary contrastive loss (SimCSE Section 6.3 generalization)
# ===========================================================================
def ordinal_infonce_loss(embeddings, labels, dist_matrix,
                         temp=0.05, alpha=2.0, metric="cosine", eps=1e-8):
    """Supervised InfoNCE with ordinal-aware (label-distance) negative weighting.

    This generalizes the *hard negatives* mechanism of SimCSE (Gao et al., 2021,
    Eq. 8 / Section 6.3).  In supervised SimCSE the per-anchor objective is

        l_i = -log [ e^{sim(h_i, h_i+)/t}
                     / sum_j ( e^{sim(h_i, h_j+)/t}
                               + alpha^{1_i^j} * e^{sim(h_i, h_j-)/t} ) ]

    where the weight is ``alpha`` *raised to* the indicator ``1_i^j`` (=1 iff
    j==i): the single true hard negative (the contradiction hypothesis) gets
    weight ``alpha`` while every other in-batch negative keeps weight
    ``alpha^0 = 1``.  Hence ``alpha = 1`` recovers the unweighted objective.

    In ordinal classification we have no contradiction pairs; instead every
    differently-labelled sample is a negative whose "hardness" is graded by the
    label distance ``d_ij = dist_matrix[y_i][y_j]``.  We therefore replace the
    binary indicator exponent with the *normalized* label distance:

        w(d) = alpha ** d_norm,   d_norm = d / (K - 1)  in [0, 1]

    This keeps the SimCSE parameterization intact and **bounded in [1, alpha]**
    (for alpha >= 1): the farthest-label negative (d_norm = 1) gets exactly
    ``alpha`` — the SimCSE hard-negative weight — while a label-adjacent negative
    (small d_norm) stays near 1.  ``alpha = 1`` makes every weight 1.0 and
    recovers plain supervised SimCSE / SupCon; ``alpha > 1`` pushes ordinally-far
    negatives apart more strongly than ordinally-near ones.

    Because the batch contains many same-label samples (batch-all), the positive
    set ``P(i)`` per anchor has more than one element; we average the log-prob over
    all positives (SupCon-style):

        l_i = -1/|P(i)| sum_{p in P(i)} log [ e^{sim(i,p)/t}
                / ( sum_{p' in P(i)} e^{sim(i,p')/t}
                    + sum_{n in N(i)} w(d_in) e^{sim(i,n)/t} ) ]

    Args:
        embeddings: (B, H) sentence embeddings (e.g. the [CLS] token).
        labels:     (B,) integer class labels.
        dist_matrix: (K, K) label-distance matrix (list or tensor); dist_matrix[i][j].
        temp:       softmax temperature ``t``.
        alpha:      SimCSE-style negative weight on the farthest label (>= 1);
                    1.0 = plain SimCSE (no ordinal weighting).
        metric:     'cosine' (default) or 'euclidean'.
    Returns:
        Scalar loss (mean over anchors that have at least one positive).
    """
    device = embeddings.device
    B = embeddings.shape[0]
    labels = labels.view(-1)

    if metric == "euclidean":
        # higher similarity == smaller distance
        sim = -torch.cdist(embeddings, embeddings)
    else:  # cosine
        z = F.normalize(embeddings, dim=1)
        sim = z @ z.t()
    sim = sim / temp

    # Positive / negative / self masks.
    eye = torch.eye(B, dtype=torch.bool, device=device)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = same_label & ~eye          # same label, not self
    neg_mask = ~same_label                # different label

    # Per-pair label distance -> SimCSE-faithful, bounded ordinal weight w = alpha**d_norm.
    dist_t = torch.as_tensor(dist_matrix, dtype=sim.dtype, device=device)  # (K, K)
    K = dist_t.shape[0]
    d = dist_t[labels][:, labels]                          # (B, B) = dist(y_i, y_j)
    d_norm = d / max(K - 1, 1)
    w = alpha ** d_norm                                    # in [1, alpha] for alpha >= 1

    # Numerically-stable exp(sim), with self-similarity removed from the denominator.
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim - sim_max) * (~eye).to(sim.dtype)

    denom = (exp_sim * pos_mask).sum(1) + (exp_sim * neg_mask * w).sum(1)  # (B,)
    log_prob = (sim - sim_max) - torch.log(denom.unsqueeze(1) + eps)       # (B, B)

    pos_count = pos_mask.sum(1)                                            # (B,)
    valid = pos_count > 0
    loss_per_anchor = -(log_prob * pos_mask).sum(1) / pos_count.clamp(min=1)
    if valid.any():
        return loss_per_anchor[valid].mean()
    return embeddings.new_zeros(())


def make_ordinal_aux_trainer(base_trainer_cls, triplet_weight=0.1,
                             triplet_temp=0.05, triplet_alpha=2.0,
                             triplet_metric="cosine"):
    """Wrap any of the loss Trainers above so it adds the ordinal InfoNCE term.

    The returned Trainer keeps ``base_trainer_cls`` as the *main* loss and adds
    ``triplet_weight * ordinal_infonce_loss(...)`` computed on the encoder's
    [CLS] embedding.  Works for every trainer here (and the plain ``Trainer``
    used by CE/CORAL) because they all return ``(loss, outputs)`` with an
    ``outputs.hidden_states`` populated once ``output_hidden_states=True`` is
    requested.
    """
    class OrdinalAuxTrainer(base_trainer_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Ask the encoder for hidden states so we can read the [CLS] embedding.
            inputs = dict(inputs)
            inputs["output_hidden_states"] = True
            main_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs)

            m = _unwrap(model)
            labels = inputs["labels"]
            cls_emb = outputs.hidden_states[-1][:, 0]  # (B, H) last-layer [CLS]
            aux = ordinal_infonce_loss(
                cls_emb, labels, m.dist_matrix,
                temp=triplet_temp, alpha=triplet_alpha, metric=triplet_metric)
            loss = main_loss + triplet_weight * aux
            return (loss, outputs) if return_outputs else loss

    OrdinalAuxTrainer.__name__ = f"Aux{base_trainer_cls.__name__}"
    return OrdinalAuxTrainer