#!/bin/bash
#SBATCH -J debug_extract_internvl2
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --mem=64G
#SBATCH -t 00:20:00

set -euo pipefail

echo "===== JOB START $(date) ====="

# ---- conda ----
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# ---- project ----
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export PROJECT_ROOT="$PROJECT_DIR"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# ---- model / data ----
MODEL_KEY="OpenGVLab/InternVL2-26B"
MODEL_TAG="internvl2-26b"

DATASET_DIR="data/probing/scene_description_balanced_2d"
OUT_DIR="data/probing/scene_description_balanced_2d_out_${MODEL_TAG}"
PROMPT_PATH="prompts/scene_description_2D_parse.txt"
LAYERS="0,10,20"

echo "=================================================="
echo "MODEL_KEY   = $MODEL_KEY"
echo "MODEL_TAG   = $MODEL_TAG"
echo "DATASET     = $DATASET_DIR"
echo "OUT_DIR     = $OUT_DIR"
echo "PROMPT_PATH = $PROMPT_PATH"
echo "LAYERS      = $LAYERS"
echo "=================================================="

# ---- create output dirs explicitly ----
mkdir -p "$OUT_DIR/tokens"

# ---- STEP 1 ONLY: extract image tokens (1 sample) ----
echo "[$(date --iso-8601=seconds)] STEP 1: Extract image tokens (max_samples=1)"

python -m probing.extract_image_tokens \
  --model_key "$MODEL_KEY" \
  --model_tag "$MODEL_TAG" \
  --dataset_dir "$DATASET_DIR" \
  --out_dir "$OUT_DIR" \
  --layers "$LAYERS" \


echo "===== JOB END $(date) ====="
