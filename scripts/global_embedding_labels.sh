#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-qwen3-vl-30b-a3b-instruct}"
DATASET_DIR="${2:-data/probing/scene_description_balanced_2d}"
TOKENS_DIR="${3:-data/probing/scene_description_balanced_2d_out_${MODEL_TAG}/tokens}"
OUT_DIR="${4:-output/probing_global/scene_description_balanced_2d_${MODEL_TAG}}"
POOL="${5:-mean}"

OUT_NPZ="${OUT_DIR}/global_embeddings.npz"
if [[ -f "$OUT_NPZ" && "${OVERWRITE:-0}" != "1" ]]; then
  echo "[global_embedding_labels] Found $OUT_NPZ; skipping (set OVERWRITE=1 to rebuild)."
  exit 0
fi

python probing/build_global_embeddings.py \
  --dataset_dir "$DATASET_DIR" \
  --tokens_dir "$TOKENS_DIR" \
  --out_dir "$OUT_DIR" \
  --pool "$POOL"
