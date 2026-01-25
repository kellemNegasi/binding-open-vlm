#!/bin/bash
#SBATCH -J probing_global_internvl2_26b
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g:1
#SBATCH --mem=64G
#SBATCH -t 10:00:00

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

# ---- config ----
MODEL_KEY="OpenGVLab/InternVL2-26B"
MODEL_TAG="internvl2-26b"
DATASET_DIR="data/probing/scene_description_balanced_2d"
LAYERS="0,10,20"

echo "=================================================="
echo "MODEL_KEY  = $MODEL_KEY"
echo "MODEL_TAG  = $MODEL_TAG"
echo "DATASET    = $DATASET_DIR"
echo "LAYERS     = $LAYERS"
echo "=================================================="

# ---- 1) extract image tokens ----
echo "[$(date --iso-8601=seconds)] 1) Extract image tokens"

python -m probing.extract_image_tokens \
  --model_key "$MODEL_KEY" \
  --model_tag "$MODEL_TAG" \
  --dataset_dir "$DATASET_DIR" \
  --out_dir "data/probing/scene_description_balanced_2d_out_${MODEL_TAG}" \
  --layers "$LAYERS"

# ---- 2) build global embeddings + labels ----
echo "[$(date --iso-8601=seconds)] 2) Build global embeddings + labels"
bash scripts/global_embedding_labels.sh \
  "$MODEL_TAG" \
  "$DATASET_DIR"

# ---- 3) train linear probes (creates intermediates/) ----
echo "[$(date --iso-8601=seconds)] 3) Train multi-label global probe"
bash scripts/train_multi_label_probe_per_sample.sh \
  "$MODEL_TAG"

echo "===== JOB END $(date) ====="
