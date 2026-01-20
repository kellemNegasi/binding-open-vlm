#!/bin/bash
# SLURM job for running the global illusory-conjunction probing pipeline.
# Runs global embedding/label build, then multi-label probe training.

#SBATCH -J probing_global_qwen3_30b
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200-141g
#SBATCH --mem=64G
#SBATCH -t 10:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export PROJECT_ROOT="$PROJECT_DIR"

MODEL_KEY="${MODEL_KEY:-qwen3-vl-30b-a3b-instruct}"
MODEL_TAG="${MODEL_TAG:-qwen3-vl-30b-a3b-instruct}"
DATASET_DIR="${DATASET_DIR:-data/probing/scene_description_balanced_2d}"
LAYERS="${LAYERS:-0,10,20}"

echo "[$(date --iso-8601=seconds)] 1) Extract image tokens (overwrite)"
bash scripts/extract_image_tokens.sh "$MODEL_KEY" "$MODEL_TAG" "$DATASET_DIR" \
  "data/probing/scene_description_balanced_2d_out_${MODEL_TAG}" \
  "prompts/scene_description_2D_parse.txt" "$LAYERS"

echo "[$(date --iso-8601=seconds)] 2) Build global embeddings + labels"
bash scripts/global_embedding_labels.sh "$MODEL_TAG" "$DATASET_DIR"

echo "[$(date --iso-8601=seconds)] 3) Train multi-label global probe"
bash scripts/train_multi_label_probe.sh "$MODEL_TAG"

echo "[$(date --iso-8601=seconds)] Done"
