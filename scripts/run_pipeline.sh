#!/usr/bin/env bash
# Full reproduction pipeline for the OLL paper experiments.
# Run from the repo root: bash scripts/run_pipeline.sh
#
# Optional env vars:
#   DATASETS   e.g. "sst5 snli"           (default: all four)
#   LOSSES     e.g. "CE OLL1 OLL2"        (default: all ten)
#   LRS        e.g. "1e-4 5e-5"           (default: 5 from paper)
#   SEEDS      e.g. "1 2 3"               (default: 1 2 3 4 5)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# "amazon_reviews" excluded: HuggingFace source defunct (DefunctDatasetError).
# Re-add it (DATASETS="amazon_reviews snli sst5 yelp") once an alternative is available.
DATASETS="${DATASETS:-snli sst5 yelp}"
LOSSES="${LOSSES:-CE OLL1 OLL15 OLL2 WKL SOFT2 SOFT3 SOFT4 EMD CORAL}"
LRS="${LRS:-1e-4 7.5e-5 5e-5 2.5e-5 1e-5}"
SEEDS="${SEEDS:-1 2 3 4 5}"

echo "======================================================="
echo " OLL Paper — Full Reproduction Pipeline"
echo "======================================================="
echo " Datasets : $DATASETS"
echo " Losses   : $LOSSES"
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

# ── Step 3: Training ────────────────────────────────────────
echo ""
echo "[3/5] Training ..."
python -m src.training \
    --datasets $DATASETS \
    --losses   $LOSSES \
    --learning_rates $LRS \
    --seeds    $SEEDS

# ── Step 4: Evaluation / Inference ──────────────────────────
echo ""
echo "[4/5] Running inference on test sets ..."
python -m src.inference --datasets $DATASETS

# ── Step 5: Analysis & Visualisation ────────────────────────
echo ""
echo "[5/5] Generating analysis and figures ..."
python -m scripts.analyze_results  --datasets $DATASETS
python -m scripts.visualize_results --datasets $DATASETS

echo ""
echo "Pipeline complete."
echo "  Metrics  : src/outputs_training/output_metrics/metrics_test_set.csv"
echo "  Figures  : results/figures/"
