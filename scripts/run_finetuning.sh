#!/usr/bin/env bash
# Classifier fine-tuning using per-dataset fine-tuned embedding as backbone.
#
# For each dataset, the best fine-tuned embedding is resolved from
# src/embedding_config.json (selected by kNN-MAE-10). The corresponding
# HuggingFace model is used as the --model_checkpoint for training.py.
#
# Learning rates are NOT grid-searched: the best per-(dataset, loss) LR
# from src/loss_config.json is used directly.
#
# Run from the repo root:
#   bash scripts/run_finetuning.sh
#
# Optional env-var overrides (space-separated lists):
#   DATASETS  e.g. "sst5 snli"    (default: all four)
#   LOSSES    e.g. "CE OLL1"      (default: all eleven)
#   SEEDS     e.g. "1 2 3"        (default: 1 2 3 4 5)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-amazon_reviews snli yelp sst5}"
LOSSES="${LOSSES:-CE BCE OLL1 OLL15 OLL2 WKL SOFT2 SOFT3 SOFT4 EMD CORAL}"
SEEDS="${SEEDS:-1}"
# 2 3 4 5
MODEL_ALIAS="bert-tiny"

# ── HuggingFace username ─────────────────────────────────────
HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null)
if [ -z "$HF_USER" ]; then
    echo "ERROR: Not logged into HuggingFace. Run 'huggingface-cli login' first."
    exit 1
fi

# ── Summary header ───────────────────────────────────────────
echo "======================================================="
echo " Classifier Fine-Tuning — Fine-Tuned Embedding Backbone"
echo "======================================================="
echo " HF user  : ${HF_USER}"
echo " Datasets : ${DATASETS}"
echo " Losses   : ${LOSSES}"
echo " Seeds    : ${SEEDS}"
echo "======================================================="

total=0
passed=0
failed=0
failed_runs=()

for DATASET in $DATASETS; do

    # Resolve best embedding config for this dataset
    read -r K_BEST MARGIN_BEST METRIC_BEST < <(python -c "
import json
with open('src/embedding_config.json') as f:
    cfg = json.load(f)
d = cfg['$DATASET']
print(d['k_proxies'], d['margin_type'], d['metric'])
")

    MODEL_ID="${MODEL_ALIAS}-${DATASET}-k${K_BEST}-${MARGIN_BEST}-${METRIC_BEST}"
    HF_MODEL="${HF_USER}/${MODEL_ID}"

    echo ""
    echo "======================================================="
    echo " Dataset  : ${DATASET}"
    echo " Backbone : ${HF_MODEL}"
    echo "======================================================="

    for LOSS in $LOSSES; do
        total=$((total + 1))

        LR=$(python -c "
import json
with open('src/loss_config.json') as f:
    cfg = json.load(f)
print(cfg['$DATASET']['$LOSS'])
")

        echo ""
        echo "-----------------------------------------------------------"
        echo "[RUN ${total}] dataset=${DATASET}  loss=${LOSS}  lr=${LR}"
        echo "  backbone: ${HF_MODEL}"
        echo "-----------------------------------------------------------"

        python -m src.training \
            --datasets         "$DATASET" \
            --losses           "$LOSS" \
            --learning_rates   "$LR" \
            --seeds            $SEEDS \
            --model_checkpoint "$HF_MODEL"

        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "[FAILED] ${DATASET}/${LOSS} (exit code ${EXIT_CODE})"
            failed=$((failed + 1))
            failed_runs+=("${DATASET}/${LOSS}")
        else
            echo "[DONE] ${DATASET}/${LOSS}"
            passed=$((passed + 1))
        fi
    done
done

# ── Final summary ────────────────────────────────────────────
echo ""
echo "======================================================="
echo " Pipeline complete"
echo "   Total  : ${total}"
echo "   Done   : ${passed}"
echo "   Failed : ${failed}"
if [ ${failed} -gt 0 ]; then
    echo ""
    echo " Failed runs:"
    for run in "${failed_runs[@]}"; do
        echo "   - ${run}"
    done
fi
echo "======================================================="
