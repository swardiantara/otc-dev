import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 1. Custom Evaluator: NOS (Neighborhood Ordinal Smoothness)
# ==========================================
class NOSEvaluator(SentenceEvaluator):
    def __init__(self, sentences, labels, k_values=[1, 3, 5, 10], name="val"):
        self.sentences = sentences
        self.labels = np.array(labels)
        self.k_values = k_values
        self.name = name

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> dict:
        logger.info(f"Running NOS Evaluation (epoch: {epoch})")
        embeddings = model.encode(self.sentences, convert_to_numpy=True, show_progress_bar=False)
        
        max_k = max(self.k_values)
        nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric='cosine').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        avg_nos = 0.0
        nos_scores = {}
        for k in self.k_values:
            k_indices = indices[:, 1:k+1]
            neighbor_labels = self.labels[k_indices]
            label_diffs = np.abs(self.labels[:, None] - neighbor_labels)
            nos_k = np.mean(label_diffs)
            nos_scores[f"NOS_{k}"] = nos_k
            avg_nos += nos_k
            
        final_nos = avg_nos / len(self.k_values)
        nos_scores["NOS_avg"] = final_nos
        logger.info(f"NOS Scores: {nos_scores}")
        
        return {f"{self.name}_NOS_avg": final_nos}

# ==========================================
# 2. Custom Loss: Adaptive Margin Contrastive
# ==========================================
class OrdinalProxyContrastiveLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, margin_type='adaptive', distance_metric='euclidean', max_margin=1.0, fixed_margin=0.5):
        super().__init__()
        self.model = model
        self.margin_type = margin_type
        self.distance_metric = distance_metric
        self.max_margin = max_margin
        self.fixed_margin = fixed_margin

    def forward(self, sentence_features, labels):
        emb1 = self.model(sentence_features[0])['sentence_embedding']
        emb2 = self.model(sentence_features[1])['sentence_embedding']
        
        is_positive = labels[:, 0].float()
        normalized_label_dist = labels[:, 1].float()
        
        if self.distance_metric == 'euclidean':
            distances = F.pairwise_distance(emb1, emb2)
        elif self.distance_metric == 'cosine':
            distances = 1 - F.cosine_similarity(emb1, emb2)
            
        if self.margin_type == 'adaptive':
            margin = normalized_label_dist * self.max_margin
        else:
            margin = torch.tensor(self.fixed_margin, device=emb1.device)
            
        # Hadsell Contrastive Formulation
        loss_pos = is_positive * torch.pow(distances, 2)
        loss_neg = (1 - is_positive) * torch.pow(torch.clamp(margin - distances, min=0.0), 2)
        
        return torch.mean(loss_pos + loss_neg)

# ==========================================
# 3. Data Loading & Medoid Selection
# ==========================================
def get_dataset_info(dataset_name):
    # Mapping standardized names to HF datasets and their text/label columns
    mapping = {
        'sst5': ('SetFit/sst5', 'text', 'label', 4),
        'snli': ('snli', 'premise', 'label', 2), # Using premise. You might want to concat premise+hypothesis
        'amazon': ('amazon_reviews', 'text', 'label', 4), # labels 1-5 mapped to 0-4
        'yelp': ('yelp_review_full', 'text', 'label', 4)
    }
    return mapping.get(dataset_name.lower())

def create_medoid_pairs(texts, labels, embeddings, k_proxies, max_label_diff):
    class_medoids = {}
    
    # 1. Medoid Selection
    for c in np.unique(labels):
        c_idx = np.where(labels == c)[0]
        c_emb = embeddings[c_idx]
        if k_proxies == 1:
            centroid = c_emb.mean(axis=0, keepdims=True)
            dists = pairwise_distances(c_emb, centroid)
            class_medoids[c] = [c_idx[np.argmin(dists)]]
        else:
            kmeans = KMeans(n_clusters=k_proxies, random_state=42).fit(c_emb)
            medoids = []
            for center in kmeans.cluster_centers_:
                dists = pairwise_distances(c_emb, center.reshape(1, -1))
                medoids.append(c_idx[np.argmin(dists)])
            class_medoids[c] = medoids

    # 2. Pair Construction
    pairs = {"text_a": [], "text_b": [], "label": []} # label: [is_positive, normalized_dist]
    
    for i, (txt, lbl) in enumerate(zip(texts, labels)):
        c_medoids = class_medoids[lbl]
        
        # Positive pair (Sample to its closest class medoid)
        if k_proxies == 1:
            closest_medoid_idx = c_medoids[0]
        else:
            dists = pairwise_distances(embeddings[i].reshape(1, -1), embeddings[c_medoids])
            closest_medoid_idx = c_medoids[np.argmin(dists)]
            
        pairs["text_a"].append(txt)
        pairs["text_b"].append(texts[closest_medoid_idx])
        pairs["label"].append([1.0, 0.0]) # Positive pair, 0 dist
        
        # Negative pairs (Sample to other classes' medoids)
        for other_c, other_medoids in class_medoids.items():
            if other_c == lbl: continue
            norm_dist = abs(lbl - other_c) / max_label_diff
            for m_idx in other_medoids:
                pairs["text_a"].append(txt)
                pairs["text_b"].append(texts[m_idx])
                pairs["label"].append([0.0, norm_dist]) # Negative pair, adaptive dist
                
    # 3. Medoid-to-Medoid Positive Pairs (Pull sub-clusters together)
    if k_proxies > 1:
        for c, medoids in class_medoids.items():
            for m1 in medoids:
                for m2 in medoids:
                    if m1 != m2:
                        pairs["text_a"].append(texts[m1])
                        pairs["text_b"].append(texts[m2])
                        pairs["label"].append([1.0, 0.0])
                        
    return Dataset.from_dict(pairs)

# ==========================================
# 4. Main Training Pipeline
# ==========================================
def main(args):
    ds_info = get_dataset_info(args.dataset)
    if not ds_info:
        raise ValueError(f"Dataset {args.dataset} not supported.")
        
    hf_path, text_col, label_col, max_label_diff = ds_info
    
    logger.info("Loading dataset...")
    raw_ds = load_dataset(hf_path)
    # Subset for speed if needed during dev
    train_texts = raw_ds['train'][text_col]
    train_labels = raw_ds['train'][label_col]
    val_texts = raw_ds['validation'][text_col]
    val_labels = raw_ds['validation'][label_col]

    # Filter out invalid labels (e.g., SNLI has -1 for missing consensus)
    valid_train = [(t, l) for t, l in zip(train_texts, train_labels) if l >= 0]
    train_texts, train_labels = zip(*valid_train)
    valid_val = [(t, l) for t, l in zip(val_texts, val_labels) if l >= 0]
    val_texts, val_labels = zip(*valid_val)

    logger.info("Initializing BERT-tiny...")
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    logger.info("Extracting embeddings for Medoid selection...")
    train_embeddings = model.encode(train_texts, convert_to_numpy=True, show_progress_bar=True)
    
    logger.info(f"Constructing pairs (k={args.k_proxies})...")
    train_dataset = create_medoid_pairs(train_texts, train_labels, train_embeddings, args.k_proxies, max_label_diff)
    
    logger.info("Setting up Evaluator and Loss...")
    evaluator = NOSEvaluator(val_texts, val_labels, name="val")
    loss = OrdinalProxyContrastiveLoss(
        model=model, 
        margin_type=args.margin_type, 
        distance_metric=args.distance_metric,
        max_margin=args.max_margin,
        fixed_margin=args.fixed_margin
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=f"./results_{args.dataset}_{args.margin_type}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_NOS_avg",
        greater_is_better=False, # Lower NOS is better
        hub_token=args.hf_token,
        push_to_hub=True if args.hub_model_id else False,
        hub_model_id=args.hub_model_id
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        evaluator=evaluator,
        loss=loss
    )

    logger.info("Starting Fine-tuning...")
    trainer.train()

    if args.hub_model_id:
        logger.info(f"Pushing best model to Hub: {args.hub_model_id}")
        trainer.push_to_hub()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ordinal Proxy Contrastive Fine-Tuning")
    parser.add_argument("--dataset", type=str, required=True, choices=['sst5', 'snli', 'amazon', 'yelp'])
    parser.add_argument("--model_name", type=str, default="google/bert_uncased_L-2_H-128_A-2")
    parser.add_argument("--k_proxies", type=int, default=1, help="Number of proxy medoids per class")
    parser.add_argument("--margin_type", type=str, choices=['adaptive', 'fixed'], default='adaptive')
    parser.add_argument("--distance_metric", type=str, choices=['euclidean', 'cosine'], default='cosine')
    parser.add_argument("--max_margin", type=float, default=1.0, help="Max margin for adaptive setting")
    parser.add_argument("--fixed_margin", type=float, default=0.5, help="Margin for fixed setting")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--hub_model_id", type=str, default=None, help="E.g., your_username/ordinal-bert-tiny")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace Write Token")
    
    args = parser.parse_args()
    main(args)