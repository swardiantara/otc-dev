#!/usr/bin/env bash
# Best-configuration pipeline for the ordinal-aware auxiliary contrastive loss.
#
# Same train -> infer -> analyze -> visualize flow as scripts/run_pipeline.sh, but:
#   * training is run WITH --add_triplet_loss, so each (dataset, loss) is fine-tuned
#     with the chosen main loss PLUS the SimCSE-style ordinal-weighted InfoNCE
#     auxiliary term (see README "Optional: ordinal-aware auxiliary contrastive
#     loss" and src/loss_functions.py:ordinal_infonce_loss);
#   * there is NO learning-rate sweep — each (dataset, loss) is trained only at its
#     best learning rate, looked up from src/loss_config.json in the loop below and
#     passed to the (unmodified) training script via --learning_rates.
#
# Runs are tagged <LOSS>-TRIPa<alpha> (e.g. OLL2-TRIPa1p5) in checkpoint/metric
# names, so they live alongside the plain-loss baselines from run_pipeline.sh and
# show up as distinct, directly comparable rows in the analysis/figures. Run the
# baseline pipeline first (or in parallel) to get the OLL2-vs-OLL2-TRIPa1p5 style
# comparison.
#
# Run from the repo root:  bash scripts/run_triplet_pipeline.sh
#
# Optional env vars:
#   DATASETS    e.g. "sst5 snli"        (default: all four)
#   LOSSES      e.g. "CE OLL1 OLL2"     (default: all eleven)
#   SEEDS       e.g. "1 2 3"            (default: 1 2 3 4 5)
#   LR_CONFIG   path to {dataset:{loss:best_lr}} JSON (default: src/loss_config.json)
#   TRIP_ALPHA  ordinal weight on the farthest label, w=alpha**d_norm in [1,alpha]
#                                       (default: 1.5; SimCSE setting is 1.0)
#   TRIP_TEMP   InfoNCE temperature tau (default: 0.05 — SimCSE's best temperature)
#   TRIP_WEIGHT lambda mixing weight, total = main + lambda*aux
#                                       (default: 0.1)
#
# Notes on the triplet hyperparameters:
#   * TRIP_TEMP=0.05 is the best temperature reported in the SimCSE paper
#     (Appendix D temperature ablation).
#   * SimCSE has no auxiliary-loss mixing weight (its contrastive loss IS the loss),
#     so TRIP_WEIGHT has no SimCSE counterpart; 0.1 is this repo's recommended
#     default for adding the term on top of an existing classification loss.
#   * Only alpha is encoded into the run tag (the axis intended for sweeping). If
#     you also vary TRIP_TEMP / TRIP_WEIGHT, extend the tag in src/training.py so
#     those runs don't overwrite each other.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-amazon_reviews snli sst5 yelp}"
LOSSES="${LOSSES:-BCE CE OLL1 OLL15 OLL2 CORAL WKL SOFT2 SOFT3 SOFT4 EMD}"
SEEDS="${SEEDS:-1 2 3 4 5}"
LR_CONFIG="${LR_CONFIG:-src/loss_config.json}"

TRIP_ALPHA="${TRIP_ALPHA:-1.5}"
TRIP_TEMP="${TRIP_TEMP:-0.05}"
TRIP_WEIGHT="${TRIP_WEIGHT:-0.1}"

echo "======================================================="
echo " OLL + Ordinal-Aware Triplet (auxiliary InfoNCE) Pipeline"
echo "======================================================="
echo " Datasets    : $DATASETS"
echo " Losses      : $LOSSES"
echo " LR          : best per (dataset, loss) from $LR_CONFIG"
echo " Seeds       : $SEEDS"
echo " triplet_alpha  : $TRIP_ALPHA"
echo " triplet_temp   : $TRIP_TEMP   (SimCSE best temperature)"
echo " triplet_weight : $TRIP_WEIGHT"
echo "======================================================="

# ── Step 1: Install dependencies ────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies ..."
pip install -r requirements.txt

# ── Step 2: Prepare datasets ────────────────────────────────
echo ""
echo "[2/5] Preparing datasets ..."
python -m scripts.prepare_datasets --datasets $DATASETS

# ── Step 3: Training (with the ordinal-aware auxiliary loss) ─
echo ""
echo "[3/5] Training with --add_triplet_loss (tag: <LOSS>-TRIPa${TRIP_ALPHA/./p}), best LR per (dataset, loss) from $LR_CONFIG ..."
# One training invocation per (dataset, loss): look up the best LR for the pair in
# $LR_CONFIG and pass it as a single --learning_rates value. The training script is
# used unchanged. Pairs missing from the config are skipped with a warning.
for ds in $DATASETS; do
    for loss in $LOSSES; do
        lr="$(python -c "import json; print(json.load(open('$LR_CONFIG'))['$ds']['$loss'])" 2>/dev/null)" || true
        if [ -z "$lr" ]; then
            echo "  ! No LR for $ds/$loss in $LR_CONFIG — skipping."
            continue
        fi
        echo "  -> $ds / $loss @ lr=$lr"
        python -m src.training \
            --datasets $ds \
            --losses   $loss \
            --learning_rates "$lr" \
            --seeds    $SEEDS \
            --add_triplet_loss \
            --triplet_alpha  "$TRIP_ALPHA" \
            --triplet_temp   "$TRIP_TEMP" \
            --triplet_weight "$TRIP_WEIGHT"
    done
done

# ── Step 4: Evaluation / Inference ──────────────────────────
# inference.py parses the -TRIPa<alpha> tag automatically and writes the full
# label (e.g. OLL2-TRIPa1p5) into the shared metrics_test_set.csv.
echo ""
echo "[4/5] Running inference on test sets ..."
python -m src.inference --datasets $DATASETS

# ── Step 5: Analysis & Visualisation ────────────────────────
# analyze/visualize group by the loss label, so the triplet runs appear next to
# their plain-loss baselines for direct comparison.
echo ""
echo "[5/5] Generating analysis and figures ..."
# Use each script's default output dir (absolute results/ and results/figures/).
python -m scripts.analyze_results   --datasets $DATASETS
python -m scripts.visualize_results --datasets $DATASETS

echo ""
echo "Pipeline complete."
echo "  Metrics  : src/outputs_training/output_metrics/<dataset>/metrics_test_set.csv"
echo "  Summary  : results/*.xlsx"
echo "  Figures  : results/figures/"
