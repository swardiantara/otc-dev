#!/usr/bin/env bash
# Constrained Proxies Learning (CPL) fine-tuning pipeline for ordinal text
# classification — runs src/finetune_cpl.py across datasets, CPL variants and
# seeds, then evaluates on the test set (metrics are written directly by the
# Python script, in the same schema as src/inference.py).
#
# Reference: "Controlling Class Layout for Deep Ordinal Classification via
#             Constrained Proxies Learning", Wang et al., AAAI 2023.
#             https://github.com/Tenvence/cpl
#
# Run from the repo root:
#   bash scripts/run_cpl.sh
#
# Optional env-var overrides (space-separated lists):
#   DATASETS    e.g. "sst5 snli"                 (default: all four)
#   CONSTRAINTS e.g. "H-L S-B"                   (default: all four CPL variants)
#   SEEDS       e.g. "1 2 3"                     (default: 1)
#   EPOCHS      e.g. "30"                        (default: 20)
#   BATCH_SIZE  e.g. "128"                       (default: 256)
#   LR          e.g. "5e-5"                      (feature extractor lr; default: 5e-5)
#   LR_PL_MUL   e.g. "10"                        (proxies-learner lr multiplier; default: 10)
#   FEATURE_DIM e.g. "512"                       (default: 512)
#   EXTRA_ARGS  e.g. "--save_model"              (passed verbatim to the python script)
#
# Notes on the CPL variants (--constraint):
#   H-L : Hard-CPL, linear layout      (Euclidean metric, forced)
#   H-S : Hard-CPL, semicircular layout (cosine metric, forced)
#   S-P : Soft-CPL, Poisson  smoothing  (metric set by --metric_method)
#   S-B : Soft-CPL, Binomial smoothing  (metric set by --metric_method)
# For the soft variants this script sweeps both Euclidean (E) and Cosine (C)
# metrics, matching the paper's ablation; the hard variants fix the metric.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASETS="${DATASETS:-amazon_reviews snli yelp sst5}"
CONSTRAINTS="${CONSTRAINTS:-H-L H-S S-P S-B}"
SEEDS="${SEEDS:-1}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-5e-5}"
LR_PL_MUL="${LR_PL_MUL:-10}"
FEATURE_DIM="${FEATURE_DIM:-512}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "======================================================="
echo " Constrained Proxies Learning (CPL) — Ordinal Text"
echo "======================================================="
echo " Datasets    : ${DATASETS}"
echo " Constraints : ${CONSTRAINTS}"
echo " Seeds       : ${SEEDS}"
echo " Epochs      : ${EPOCHS}   Batch: ${BATCH_SIZE}   Feature dim: ${FEATURE_DIM}"
echo " LR (feat)   : ${LR}   (proxies lr = LR x ${LR_PL_MUL})"
echo "======================================================="

total=0; passed=0; failed=0
failed_runs=()

run_one() {
    local dataset="$1" constraint="$2" metric="$3"
    total=$((total + 1))

    echo ""
    echo "-----------------------------------------------------------"
    echo "[RUN ${total}] dataset=${dataset}  constraint=${constraint}  metric=${metric}"
    echo "-----------------------------------------------------------"

    python -m src.finetune_cpl \
        --dataset        "$dataset" \
        --constraint     "$constraint" \
        --metric_method  "$metric" \
        --feature_dim    "$FEATURE_DIM" \
        --epochs         "$EPOCHS" \
        --batch_size     "$BATCH_SIZE" \
        --lr             "$LR" \
        --lr_pl_mul      "$LR_PL_MUL" \
        --seeds          $SEEDS \
        $EXTRA_ARGS

    if [ $? -ne 0 ]; then
        echo "[FAILED] ${dataset}/${constraint}/${metric}"
        failed=$((failed + 1))
        failed_runs+=("${dataset}/${constraint}/${metric}")
    else
        echo "[DONE] ${dataset}/${constraint}/${metric}"
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
            H-L|H-S)
                # Hard constraints fix the metric internally; metric flag unused.
                run_one "$DATASET" "$CONSTRAINT" "E"
                ;;
            S-P|S-B)
                # Soft constraints: sweep both Euclidean and Cosine metrics.
                run_one "$DATASET" "$CONSTRAINT" "E"
                run_one "$DATASET" "$CONSTRAINT" "C"
                ;;
            *)
                echo "Unknown constraint '${CONSTRAINT}', skipping."
                ;;
        esac
    done
done

echo ""
echo "======================================================="
echo " CPL pipeline complete"
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
echo " Test metrics -> src/outputs_training/output_metrics/<dataset>/metrics_test_set.csv"
echo " Aggregate    -> python -m scripts.analyze_results --datasets ${DATASETS}"
echo "======================================================="
