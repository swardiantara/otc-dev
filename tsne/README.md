# t-SNE representation plots (validation set)

This folder is **tracked in git** (unlike `results/` and `src/outputs_training/`,
which are ignored) so the learned-representation plots are easy to push.

`src/training.py` writes one plot per trained model after fine-tuning:

```
tsne/<dataset>/<run_name>_val.pdf
```

Each plot is a 2-D t-SNE of the **validation-set** `[CLS]` embeddings, colored by
ordinal label. `<run_name>` matches the checkpoint directory name, so baseline
and `+Aux` runs are directly comparable, e.g.:

```
tsne/sst5/bert_uncased_L-2_H-128_A-2-sst5-OLL2-1_..._val.pdf            # baseline
tsne/sst5/bert_uncased_L-2_H-128_A-2-sst5-OLL2-TRIPa1p5-1_..._val.pdf   # +Aux, alpha=1.5
```

Control via `--tsne_max_samples` (default 5000 stratified, `0` = all) and
`--no_tsne` to disable. Test-set t-SNE plots are written separately, next to the
other per-model inference artifacts under
`src/outputs_training/output_inference/` (git-ignored).
