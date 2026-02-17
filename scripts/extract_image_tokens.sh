#!/bin/bash
set -euo pipefail

MODEL_KEY="${1:-qwen3_vl_30b}"
MODEL_TAG="${2:-qwen3-vl-30b-a3b-instruct}"
DATASET_DIR="${3:-data/probing/scene_description_balanced_2d}"
OUT_DIR="${4:-data/probing/scene_description_balanced_2d_out_${MODEL_TAG}}"
PROMPT_PATH="${5:-prompts/scene_description_2D_parse.txt}"
LAYERS="${6:-0,10,20}"
DEVICE="${7:-auto}"
DTYPE="${8:-auto}"
SAVE_MODE="${9:-tokens}"
SAVE_DTYPE="${10:-float32}"

python probing/extract_image_tokens.py \
  --dataset_dir "$DATASET_DIR" \
  --out_dir "$OUT_DIR" \
  --model "$MODEL_KEY" \
  --prompt_path "$PROMPT_PATH" \
  --layers "$LAYERS" \
  --save "$SAVE_MODE" \
  --save_dtype "$SAVE_DTYPE" \
  --device "$DEVICE" --dtype "$DTYPE" \
  $( [[ "${OVERWRITE:-0}" == "1" ]] && echo "--overwrite" )
