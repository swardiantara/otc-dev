#!/usr/bin/env bash
# Grid search for ordinal proxy contrastive embedding fine-tuning.
# Run from the repo root: bash scripts/run_embedding_gridsearch.sh
#
# All margin values are given in cosine-distance space. The Python script
# converts them to the equivalent euclidean margins automatically when
# --distance_metric euclidean is selected (d_euc = sqrt(2 * d_cos)).
#
# Optional env-var overrides (space-separated lists):
#   DATASETS        e.g. "sst5 snli"             (default: all four)
#   K_PROXIES       e.g. "1 3"                   (default: 1 3 5 10)
#   MARGIN_TYPES    e.g. "adaptive"              (default: adaptive fixed)
#   DISTANCE_METRICS e.g. "cosine"               (default: cosine euclidean)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Configurable grid ────────────────────────────────────────
DATASETS="${DATASETS:-sst5 snli amazon_reviews yelp}"
K_PROXIES="${K_PROXIES:-1 3 5}"
MARGIN_TYPES="${MARGIN_TYPES:-fixed adaptive}"
DISTANCE_METRICS="${DISTANCE_METRICS:-cosine}"

# ── Fixed hyperparameters ────────────────────────────────────
MODEL_NAME="google/bert_uncased_L-2_H-128_A-2"
MODEL_ALIAS="bert-tiny"
MAX_MARGIN=1.0      # cosine space; auto-converted when euclidean is used
FIXED_MARGIN=0.5    # cosine space; auto-converted when euclidean is used
EPOCHS=10
BATCH_SIZE=1024
LEARNING_RATE="2e-5"
EARLY_STOPPING_PATIENCE=2

# ── Derived paths ────────────────────────────────────────────
METRICS_BASE="${REPO_ROOT}/results/embedding"

# ── Summary header ───────────────────────────────────────────
echo "======================================================="
echo " Contrastive Embedding Fine-Tuning — Grid Search"
echo "======================================================="
echo " Model     : ${MODEL_ALIAS} (${MODEL_NAME})"
echo " Datasets  : ${DATASETS}"
echo " k_proxies : ${K_PROXIES}"
echo " Margins   : ${MARGIN_TYPES}  (cosine max=${MAX_MARGIN}, fixed=${FIXED_MARGIN})"
echo " Metrics   : ${DISTANCE_METRICS}"
echo " Epochs    : ${EPOCHS}  LR: ${LEARNING_RATE}  Batch: ${BATCH_SIZE}  EarlyStop: ${EARLY_STOPPING_PATIENCE}"
echo "======================================================="

total=0
skipped=0
passed=0
failed=0
failed_runs=()

for K in $K_PROXIES; do
    for METRIC in $DISTANCE_METRICS; do
        for MARGIN_TYPE in $MARGIN_TYPES; do
            for DATASET in $DATASETS; do
                total=$((total + 1))

                MODEL_ID="${MODEL_ALIAS}-${DATASET}-k${K}-${MARGIN_TYPE}-${METRIC}"
                METRICS_FILE="${METRICS_BASE}/${DATASET}/${MODEL_ID}.csv"

                if [ -f "$METRICS_FILE" ]; then
                    echo "[SKIP ${total}] ${MODEL_ID}"
                    skipped=$((skipped + 1))
                    continue
                fi

                echo ""
                echo "-----------------------------------------------------------"
                echo "[RUN ${total}] dataset=${DATASET}  k=${K}  margin=${MARGIN_TYPE}  metric=${METRIC}"
                echo "-----------------------------------------------------------"

                python -m src.finetune_embedding \
                    --dataset        "$DATASET" \
                    --model_name     "$MODEL_NAME" \
                    --model_alias    "$MODEL_ALIAS" \
                    --k_proxies      "$K" \
                    --margin_type    "$MARGIN_TYPE" \
                    --distance_metric "$METRIC" \
                    --max_margin     "$MAX_MARGIN" \
                    --fixed_margin   "$FIXED_MARGIN" \
                    --epochs         "$EPOCHS" \
                    --batch_size     "$BATCH_SIZE" \
                    --learning_rate  "$LEARNING_RATE" \
                    --early_stopping_patience "$EARLY_STOPPING_PATIENCE"

                EXIT_CODE=$?
                if [ $EXIT_CODE -ne 0 ]; then
                    echo "[FAILED] ${MODEL_ID} (exit code ${EXIT_CODE})"
                    failed=$((failed + 1))
                    failed_runs+=("$MODEL_ID")
                else
                    echo "[DONE] ${MODEL_ID}"
                    passed=$((passed + 1))
                fi
            done
        done
    done
done

# # ── SST5 full-pair construction (exhaustive sample-sample pairs) ─────────────
# # Runs only after all proxy-based experiments so proxy results are available
# # for direct comparison. k_proxies is irrelevant here (not part of model_id).
# echo ""
# echo "======================================================="
# echo " SST5 full-pair construction"
# echo "======================================================="

# for METRIC in $DISTANCE_METRICS; do
#     for MARGIN_TYPE in $MARGIN_TYPES; do
#         total=$((total + 1))

#         MODEL_ID="${MODEL_ALIAS}-sst5-full-${MARGIN_TYPE}-${METRIC}"
#         METRICS_FILE="${METRICS_BASE}/sst5/${MODEL_ID}.csv"

#         if [ -f "$METRICS_FILE" ]; then
#             echo "[SKIP ${total}] ${MODEL_ID}"
#             skipped=$((skipped + 1))
#             continue
#         fi

#         echo ""
#         echo "-----------------------------------------------------------"
#         echo "[RUN ${total}] dataset=sst5  pair_mode=full  margin=${MARGIN_TYPE}  metric=${METRIC}"
#         echo "-----------------------------------------------------------"

#         python -m src.finetune_embedding \
#             --dataset        sst5 \
#             --model_name     "$MODEL_NAME" \
#             --model_alias    "$MODEL_ALIAS" \
#             --pair_mode      full \
#             --margin_type    "$MARGIN_TYPE" \
#             --distance_metric "$METRIC" \
#             --max_margin     "$MAX_MARGIN" \
#             --fixed_margin   "$FIXED_MARGIN" \
#             --epochs         "$EPOCHS" \
#             --batch_size     "$BATCH_SIZE" \
#             --learning_rate  "$LEARNING_RATE" \
#             --early_stopping_patience "$EARLY_STOPPING_PATIENCE"

#         EXIT_CODE=$?
#         if [ $EXIT_CODE -ne 0 ]; then
#             echo "[FAILED] ${MODEL_ID} (exit code ${EXIT_CODE})"
#             failed=$((failed + 1))
#             failed_runs+=("$MODEL_ID")
#         else
#             echo "[DONE] ${MODEL_ID}"
#             passed=$((passed + 1))
#         fi
#     done
# done

# ── Final summary ────────────────────────────────────────────
echo ""
echo "======================================================="
echo " Grid search complete"
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
