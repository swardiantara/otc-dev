# Ordinal Log Loss - A simple loss function for Ordinal Classification

This is the GitHub repository for the paper [A simple log-based loss function for ordinal text classification](https://aclanthology.org/2022.coling-1.407/) accepted at COLING 2022.

## Paper Abstract
The cross-entropy loss function is widely used and generally considered the default loss function for text classification. When it comes to ordinal text classification where there is an ordinal relationship between labels, the cross-entropy is not optimal as it does not incorporate the ordinal character into its feedback. In this paper, we propose a new simple loss function called ordinal log-loss (OLL). We show that this loss function outperforms state-of-the-art previously introduced losses on four benchmark text classification datasets. 


**This repository contains all the python code used to conduct the experiments reported in the paper.**

---
## Losses

In the paper, we introduce a new loss called the Ordinal Log Loss (OLL). We show that this loss, in addition to being very simple, is particularly suited for classification tasks where labels are more or less close to each other (e.g. Movie review rating classification). 

For a N classes classification task, we define the L<sub>OLL-&alpha;</sub> loss (with &alpha; is a tuneable parameter)

<img src="https://render.githubusercontent.com/render/math?math=\Large\color{grey}\textbf{\mathcal{L}_{OLL-\alpha}(P,y) = -\sum_{i=1}^{N}\log(1-p_i) d(y,i)^\alpha}">
where P = (p<sub>1</sub>, ..., p<sub>N</sub> ) is the output probability distribution of a network for a given prediction and d(y,i) is the distance between the true class y and the class i.

We compare this loss to 5 other losses :
* [Cross Entropy Loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) (CE)
* [Weighted Kappa Loss](https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666?via%3Dihub) (WKL)
* [Soft Labels Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html) (SOFT)
* [Earth Mover Distance Loss](https://arxiv.org/abs/1611.05916) (EMD)
* [Coral Loss](https://github.com/Raschka-research-group/coral-cnn) (CORAL)

The losses used in the experiments have been coded in pytorch and can be found in the file : `src/loss_functions.py`

## Datasets 

The experiments were done on 4 public datasets : 
* **[SNLI](https://nlp.stanford.edu/projects/snli/)** (Stanford Natural Language Inference): The Stanford Natural Language Inference (SNLI) corpus (version 1.0) is a collection of 570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment, contradiction, and neutral. 
* **[SST-5](https://nlp.stanford.edu/sentiment/)** : Sentiment classification of sentences extracted from movie reviews. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.
* **[Amazon Reviews](https://registry.opendata.aws/amazon-reviews-ml/)** : Sentiment classification of customer reviews on the Amazon website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.
* **[Yelp Reviews](https://www.yelp.com/dataset)** : Sentiment classification of sentences extracted from the Yelp website. Each sentence is labelled as either negative, somewhat negative, neutral, somewhat positive or positive.


## Training

### Pre-trained model
The model used in our experiments is the [google/bert_uncased_L-2_H-128_A-2](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2) which is a tiny version of the BERT model. This model can be fetched directly from the HuggingFace Model Hub.

### Reproducibility
All our experiments can be reproduced. To do so, follow steps:

1. **Download datasets**

All the datasets mentioned above can be downloaded from the [Hugging Face Datasets Hub](https://huggingface.co/datasets). 
Once downloaded, edit the `src/datasets.json` with the corresponding path for each dataset:
```
"amazon_reviews": {
        "dist": [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 0, 1, 2], [3, 2, 1, 0, 1], [4, 3, 2, 1, 0]], 
        "int2label": [1, 2, 3, 4, 5], 
        "num_classes": 5, 
        "task": ["text", null], 
        "tok_len": 128,
        "n_distances": 5,
        "path" : "COMPLETE WITH THE PATH OF DIRECTORY WITH NAME 'amazon_reviews' CONTAINING TRAIN,VALIDATION AND TEST FILES"
    }
```
For the *Amazon Reviews Dataset*, you should have a folder named `amazon_reviews` with in it the 3 following files: `amazon_reviews_train.csv`, `amazon_reviews_test.csv` and `amazon_reviews_validation.csv`.

2. Training

To train the model on the different parameters and loss functions introduced in our paper, run the `src/training.py` Python script. 

The losses used for training: 
```
losses_dict = {"CE": Trainer,
               "OLL1": OLL1Trainer,
               "OLL15": OLL15Trainer,
               "OLL2": OLL2Trainer,
               "WKL": WKLTrainer,
               "SOFT2": SOFT2Trainer,
               "SOFT3": SOFT3Trainer,
               "SOFT4": SOFT4Trainer,
               "EMD": EMDTrainer,
               "CORAL": Trainer}
```
are defined in the `src/loss_functions.py` file. 

#### Optional: ordinal-aware auxiliary contrastive loss

Any of the losses above can be combined with an **ordinal-aware contrastive auxiliary loss**, generalizing the *hard negatives* mechanism of [SimCSE](https://aclanthology.org/2021.emnlp-main.552/) (Eq. 8 / Section 6.3). Supervised SimCSE weights the single contradiction hard-negative by `alpha^indicator` (`alpha` raised to an indicator that is 1 only for the true hard negative, 0 otherwise), so `alpha=1` recovers the unweighted objective. We replace that binary indicator with the **normalized label distance** `d(y_i, y_j)` from the dataset's distance matrix:

```
w(d) = alpha ** d_norm,   d_norm = d / (num_classes - 1)  in [0, 1]
```

This keeps SimCSE's parameterization and stays **bounded in `[1, alpha]`** (cosine-compatible): the farthest-label negative (e.g. label 0 vs 4) gets exactly weight `alpha`, while a label-adjacent negative (0 vs 1) stays near 1 — so ordinally-far samples are pushed apart more strongly. The auxiliary term is a supervised InfoNCE on the `[CLS]` embedding (cosine, batch-all over positives/negatives), added on top of the chosen main loss:

```
total_loss = main_loss + triplet_weight * ordinal_infonce_loss
```

Enable it with `--add_triplet_loss`; when the flag is omitted only the chosen loss is used. `alpha = 1` recovers plain supervised SimCSE (uniform negative weights, no ordinal signal).

```bash
python -m src.training --losses OLL2 --add_triplet_loss \
    --triplet_weight 0.1 --triplet_temp 0.05 --triplet_alpha 2.0
```

| flag | default | meaning |
|------|---------|---------|
| `--add_triplet_loss` | off | enable the auxiliary loss |
| `--triplet_weight` | 0.1 | λ mixing coefficient of the auxiliary term |
| `--triplet_temp` | 0.05 | InfoNCE softmax temperature τ |
| `--triplet_alpha` | 2.0 | SimCSE weight `w=alpha**d_norm` on the farthest label (>=1; 1 = plain SimCSE) |

Auxiliary-loss runs are tagged (`<LOSS>-TRIPa<alpha>`, e.g. `OLL2-TRIPa1p5`) in checkpoint/metric names so they don't collide with the plain single-loss runs, and so different `alpha` settings coexist as distinct, comparable rows in the analysis. The inference and analysis scripts parse this tag automatically, so `scripts/run_triplet_pipeline.sh` plugs into the same train → infer → analyze → visualize flow as `scripts/run_pipeline.sh`.

3. Evaluation

Run the `scr/inference.py` file to evaluate the the model checkpoints generated during the training phase (corresponding to the different losses and parameters). It will output a csv file `src/outputs_training/output_metrics/metrics_test_set.csv` with all metrics introduced in our paper on the test sets. 

#### Per-model inference artifacts & skip mechanism

Like training (which skips a `(dataset, loss, lr, seed)` combo when its checkpoint already exists), inference now skips work that is already done. Each evaluated model gets its own artifact folder mirroring the training layout:

```
src/outputs_training/output_inference/<dataset>/<run_name>/
    metrics.json          # all scalar test metrics
    probabilities.npy     # raw per-sample prediction probabilities
    confusion_matrix.csv  # K x K confusion matrix
    tsne_test.pdf         # t-SNE of the test set ([CLS]), colored by label
    tsne_test_ob1.pdf     # same, restricted to OB1-correct samples (|pred - true| <= 1)
    _SUCCESS              # written last; its presence makes inference skip this run
```

A run is skipped when its `_SUCCESS` marker exists; pass `--force` to re-run anyway. The shared `metrics_test_set.csv` is still appended (once per model) for `analyze_results.py` / `visualize_results.py`.

#### Representation visualization (t-SNE)

To inspect the learned representations across losses (baseline vs `+Aux` with `alpha` = 1 / 1.5 / 2):

- **During training**, a validation-set t-SNE of the `[CLS]` embeddings is saved per model to the git-tracked `tsne/<dataset>/<run_name>_val.pdf` (see `tsne/README.md`).
- **During inference**, the test-set t-SNE (all samples) and an OB1-correct-only t-SNE are saved alongside the other inference artifacts above.

Both stages accept `--tsne_max_samples N` (stratified per-class cap, default 5000; `0` = all samples) and `--no_tsne` to disable.

**Note**: In `src/model_coral.py` we reimplemented the coral method as presented [here](https://github.com/Raschka-research-group/coral-cnn). 

## Citation
If you found our code useful for your research, please consider citing it:
```
@inproceedings{castagnos-etal-2022-simple,
    title = "A Simple Log-based Loss Function for Ordinal Text Classification",
    author = "Castagnos, Fran{\c{c}}ois  and
      Mihelich, Martin  and
      Dognin, Charles",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.407",
    pages = "4604--4609",
    abstract = "The cross-entropy loss function is widely used and generally considered the default loss function for text classification. When it comes to ordinal text classification where there is an ordinal relationship between labels, the cross-entropy is not optimal as it does not incorporate the ordinal character into its feedback. In this paper, we propose a new simple loss function called ordinal log-loss (OLL). We show that this loss function outperforms state-of-the-art previously introduced losses on four benchmark text classification datasets.",
}
```
