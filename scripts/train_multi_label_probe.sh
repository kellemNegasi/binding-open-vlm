#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-qwen3-vl-30b-a3b-instruct}"
OUT_DIR="${2:-output/probing_global/scene_description_balanced_2d_${MODEL_TAG}}"
EMBEDDINGS_PATH="${3:-${OUT_DIR}/global_embeddings.npz}"
TEST_SPLIT="${4:-0.2}"
SEED="${5:-0}"
C_VAL="${6:-1.0}"
THRESHOLD="${7:-0.5}"

python probing/train_global_probe.py \
  --embeddings_path "$EMBEDDINGS_PATH" \
  --out_dir "$OUT_DIR/probes" \
  --test_split "$TEST_SPLIT" --seed "$SEED" --C "$C_VAL" --threshold "$THRESHOLD"
