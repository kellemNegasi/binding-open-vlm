#!/bin/bash
# SLURM job for running the global illusory-conjunction probing pipeline.
# Runs global embedding/label build, then multi-label probe training.

#SBATCH -J probing_global_internvl2-26b_persample
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --mem=64G
#SBATCH -t 13:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export PROJECT_ROOT="$PROJECT_DIR"
export OMP_NUM_THREADS=1

MODEL_KEY="${MODEL_KEY:-OpenGVLab/InternVL2-26B}"
MODEL_TAG="${MODEL_TAG:-internvl2-26b}"
DATASET_DIR="${DATASET_DIR:-data/probing/scene_description_balanced_2d}"
LAYERS="${LAYERS:-all}"
OVR_N_JOBS="${OVR_N_JOBS:-4}"
FEATURE_JOBS="${FEATURE_JOBS:-4}"

echo "[$(date --iso-8601=seconds)] 1) Extract image tokens (resume-safe; set OVERWRITE=1 to force)"
bash scripts/extract_image_tokens.sh "$MODEL_KEY" "$MODEL_TAG" "$DATASET_DIR" \
  "data/probing/scene_description_balanced_2d_out_${MODEL_TAG}" \
  "prompts/scene_description_2D_parse.txt" "$LAYERS"

echo "[$(date --iso-8601=seconds)] 2) Build global embeddings + labels"
bash scripts/global_embedding_labels.sh "$MODEL_TAG" "$DATASET_DIR"

echo "[$(date --iso-8601=seconds)] 3) Train multi-label global probe (per-sample + intermediates)"
bash scripts/train_multi_label_probe_per_sample.sh "$MODEL_TAG"

echo "[$(date --iso-8601=seconds)] Done"
