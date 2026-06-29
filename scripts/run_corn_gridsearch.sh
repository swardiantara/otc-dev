#!/usr/bin/env bash
# Learning-rate grid search for the CORN loss on all datasets.
#
# CORN (Shi, Cao & Raschka, 2021) is a rank-consistent extension of CORAL that
# drops CORAL's weight-sharing constraint: it uses a standard head with K-1
# logits and enforces rank consistency through conditional probabilities
# (P(y>k) = prod_{j<=k} sigmoid(logit_j)) rather than the architecture.
#
# This sweeps the learning rate for CORN only, then runs the usual
# infer -> analyze -> visualize steps so the results sit alongside the other
# losses. Run from the repo root:  bash scripts/run_corn_gridsearch.sh
#
# Optional env vars:
#   DATASETS  e.g. "sst5 snli"   (default: all four)
#   LRS       learning rates to sweep
#             (default: spans the standard paper grid plus the higher range
#              where the sibling CORAL loss tended to peak)
#   SEEDS     e.g. "1 2 3"       (default: 1 2 3 4 5)
#
# Note: CORAL's best LRs in src/loss_config.json are high (1e-3 .. 5e-3), so the
# default grid below extends above the 1e-5..1e-4 paper range to give CORN a fair
# chance to find its optimum. Override LRS to narrow/widen it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-amazon_reviews snli sst5 yelp}"
LRS="${LRS:-5e-3 1e-3 5e-4 1e-4 7.5e-5 5e-5 2.5e-5 1e-5}"
SEEDS="${SEEDS:-1 2 3 4 5}"

echo "======================================================="
echo " CORN — Learning-Rate Grid Search"
echo "======================================================="
echo " Datasets : $DATASETS"
echo " Loss     : CORN"
echo " LRs      : $LRS"
echo " Seeds    : $SEEDS"
echo "======================================================="

# ── Step 1: Install dependencies ────────────────────────────
echo ""
echo "[1/5] Installing Python dependencies ..."
pip install -r requirements.txt

# ── Step 2: Prepare datasets ────────────────────────────────
echo ""
echo "[2/5] Preparing datasets ..."
python -m scripts.prepare_datasets --datasets $DATASETS

# ── Step 3: Training (CORN, LR sweep) ───────────────────────
echo ""
echo "[3/5] Training CORN across the LR grid ..."
python -m src.training \
    --datasets $DATASETS \
    --losses   CORN \
    --learning_rates $LRS \
    --seeds    $SEEDS

# ── Step 4: Evaluation / Inference ──────────────────────────
echo ""
echo "[4/5] Running inference on test sets ..."
python -m src.inference --datasets $DATASETS

# ── Step 5: Analysis & Visualisation ────────────────────────
echo ""
echo "[5/5] Generating analysis and figures ..."
python -m scripts.analyze_results   --datasets $DATASETS
python -m scripts.visualize_results --datasets $DATASETS

echo ""
echo "Grid search complete."
echo "  Metrics  : src/outputs_training/output_metrics/<dataset>/metrics_test_set.csv"
echo "  Summary  : results/*.xlsx  (best LR per (dataset, loss))"
echo "  Figures  : results/figures/"
