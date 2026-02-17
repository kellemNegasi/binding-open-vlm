#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-qwen3-vl-30b-a3b-instruct}"
OUT_DIR="${2:-output/probing_global/scene_description_balanced_2d_${MODEL_TAG}}"
EMBEDDINGS_PATH="${3:-${OUT_DIR}/global_embeddings.npz}"
TEST_SPLIT="${4:-0.2}"
SEEDS="${5:-0,1,2,3,4}"
C_VAL="${6:-1.0}"
THRESHOLD="${7:-0.5}"
OVR_N_JOBS="${OVR_N_JOBS:-1}"
FEATURE_JOBS="${FEATURE_JOBS:-1}"
PROGRESS_STEPS="${PROGRESS_STEPS:-0}"

python probing/train_global_probe.py \
  --embeddings_path "$EMBEDDINGS_PATH" \
  --out_dir "$OUT_DIR/probes" \
  --test_split "$TEST_SPLIT" \
  --seeds "$SEEDS" \
  --C "$C_VAL" \
  --threshold "$THRESHOLD" \
  --ovr_n_jobs "$OVR_N_JOBS" \
  --feature_jobs "$FEATURE_JOBS" \
  --save_intermediates \
  $( [[ "${RESUME:-0}" == "1" ]] && echo "--resume" ) \
  $( [[ "$PROGRESS_STEPS" == "1" ]] && echo "--progress_steps" )
