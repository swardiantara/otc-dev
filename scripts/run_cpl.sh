#!/usr/bin/env bash
# Constrained Proxies Learning (CPL) — EMBEDDING fine-tuning sweep.
#
# This is the embedding stage only (analogous to run_embedding_gridsearch.sh):
# it fine-tunes BERT-tiny with each CPL variant, evaluates the embedding with
# NOS + kNN-MAE on the validation set, and pushes the fine-tuned backbone to the
# HuggingFace Hub. It then runs scripts/recap_cpl.py to pick the best variant
# per dataset (by validation kNN-MAE) into src/cpl_embedding_config.json.
#
# The CLASSIFIER stage is intentionally separate: feed the selected backbone to
# src/training.py exactly as scripts/run_finetuning.sh feeds the contrastive
# embeddings (see the guidance printed at the end of this script).
#
# Reference: Wang et al., "Controlling Class Layout for Deep Ordinal
#            Classification via Constrained Proxies Learning", AAAI 2023.
#            https://github.com/Tenvence/cpl
#
# Run from the repo root:  bash scripts/run_cpl.sh
#
# Optional env-var overrides (space-separated lists):
#   DATASETS    e.g. "sst5 snli"        (default: all four)
#   CONSTRAINTS e.g. "H-L S-B"          (default: H-L H-S S-P S-B)
#   EPOCHS      e.g. "20"               (default: 10)
#   BATCH_SIZE  e.g. "128"              (default: 256)
#   LR          e.g. "5e-5"             (feature-extractor lr; default: 5e-5)
#   LR_PL_MUL   e.g. "10"               (proxies-learner lr multiplier; default: 10)
#   PUSH        "0" to skip Hub push    (default: push enabled)
#   EXTRA_ARGS  passed verbatim to the python script (e.g. "--save_model")
#
# CPL variants:
#   H-L : Hard, linear layout       (Euclidean metric, forced)
#   H-S : Hard, semicircular layout (cosine metric, forced)
#   S-P : Soft, Poisson smoothing   (metric swept: Euclidean + cosine)
#   S-B : Soft, Binomial smoothing  (metric swept: Euclidean + cosine)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-amazon_reviews snli yelp sst5}"
CONSTRAINTS="${CONSTRAINTS:-H-L H-S S-P S-B}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-5e-5}"
LR_PL_MUL="${LR_PL_MUL:-10}"
PUSH="${PUSH:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
MODEL_ALIAS="bert-tiny"

PUSH_FLAG="--push_to_hub"
[ "$PUSH" = "0" ] && PUSH_FLAG="--no_push"

METRICS_BASE="${REPO_ROOT}/results/cpl_embedding"

echo "======================================================="
echo " Constrained Proxies Learning (CPL) — Embedding Stage"
echo "======================================================="
echo " Model       : ${MODEL_ALIAS}"
echo " Datasets    : ${DATASETS}"
echo " Constraints : ${CONSTRAINTS}"
echo " Epochs      : ${EPOCHS}   Batch: ${BATCH_SIZE}"
echo " LR (feat)   : ${LR}   (proxies lr = LR x ${LR_PL_MUL})"
echo " Push to Hub : $([ "$PUSH" = "0" ] && echo no || echo yes)"
echo "======================================================="

total=0; skipped=0; passed=0; failed=0
failed_runs=()

run_one() {
    local dataset="$1" constraint="$2" metric="$3"
    total=$((total + 1))

    local model_id="${MODEL_ALIAS}-${dataset}-cpl-${constraint}-${metric}"
    local metrics_file="${METRICS_BASE}/${dataset}/${model_id}.csv"
    if [ -f "$metrics_file" ]; then
        echo "[SKIP ${total}] ${model_id}"
        skipped=$((skipped + 1))
        return
    fi

    echo ""
    echo "-----------------------------------------------------------"
    echo "[RUN ${total}] ${model_id}"
    echo "-----------------------------------------------------------"

    python -m src.finetune_cpl \
        --dataset       "$dataset" \
        --constraint    "$constraint" \
        --metric_method "$metric" \
        --model_alias   "$MODEL_ALIAS" \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH_SIZE" \
        --lr            "$LR" \
        --lr_pl_mul     "$LR_PL_MUL" \
        $PUSH_FLAG $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "[FAILED] ${model_id}"
        failed=$((failed + 1))
        failed_runs+=("$model_id")
    else
        echo "[DONE] ${model_id}"
        passed=$((passed + 1))
    fi
}

for DATASET in $DATASETS; do
    echo ""
    echo "======================================================="
    echo " Dataset : ${DATASET}"
    echo "======================================================="

    for CONSTRAINT in $CONSTRAINTS; do
        case "$CONSTRAINT" in
            H-L) run_one "$DATASET" "H-L" "E" ;;          # Euclidean, forced
            H-S) run_one "$DATASET" "H-S" "C" ;;          # cosine, forced
            S-P|S-B)                                       # sweep both metrics
                run_one "$DATASET" "$CONSTRAINT" "E"
                run_one "$DATASET" "$CONSTRAINT" "C"
                ;;
            *) echo "Unknown constraint '${CONSTRAINT}', skipping." ;;
        esac
    done
done

echo ""
echo "======================================================="
echo " Embedding sweep complete"
echo "   Total   : ${total}"
echo "   Done    : ${passed}"
echo "   Skipped : ${skipped}"
echo "   Failed  : ${failed}"
if [ ${failed} -gt 0 ]; then
    echo ""
    echo " Failed runs:"
    for run in "${failed_runs[@]}"; do
        echo "   - ${run}"
    done
fi
echo "======================================================="

# ── Recap: pick the best CPL embedding per dataset (by kNN-MAE) ──────────────
echo ""
echo "[recap] Selecting best CPL embedding per dataset ..."
python -m scripts.recap_cpl

echo ""
echo "======================================================="
echo " Next: CLASSIFIER stage (separate, like run_finetuning.sh)"
echo "======================================================="
echo " The best backbone per dataset is in src/cpl_embedding_config.json."
echo " For each dataset, train the classifier on the selected CPL backbone:"
echo ""
echo '   HF_USER=$(python -c "from huggingface_hub import whoami; print(whoami()['"'"'name'"'"'])")'
echo '   MODEL_ID=$(python -c "import json;print(json.load(open('"'"'src/cpl_embedding_config.json'"'"'))['"'"'sst5'"'"']['"'"'model_id'"'"'])")'
echo '   python -m src.training --datasets sst5 --losses CE --model_checkpoint "${HF_USER}/${MODEL_ID}"'
echo "======================================================="
