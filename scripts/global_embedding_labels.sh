#!/bin/bash
set -euo pipefail

MODEL_TAG="${1:-qwen3-vl-30b-a3b-instruct}"
DATASET_DIR="${2:-data/probing/scene_description_balanced_2d}"
TOKENS_DIR="${3:-data/probing/scene_description_balanced_2d_out_${MODEL_TAG}/tokens}"
OUT_DIR="${4:-output/probing_global/scene_description_balanced_2d_${MODEL_TAG}}"
POOL="${5:-mean}"

python probing/build_global_embeddings.py \
  --dataset_dir "$DATASET_DIR" \
  --tokens_dir "$TOKENS_DIR" \
  --out_dir "$OUT_DIR" \
  --pool "$POOL"
