#!/bin/bash
# SLURM job for running the probing pipeline (masked dataset -> token extraction -> pooling -> linear probes).
# Customize SBATCH directives below to match your quota and runtime needs.
# Default model: Qwen3-VL-30B-A3B-Instruct (Hydra key: qwen3_vl_30b).

#SBATCH -J probing_qwen3_vl_30b
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

# Optional: point caches to $SCRATCH (faster + avoids filling home).
# SCRATCH_DIR="${SCRATCH:-$HOME}"
# export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
# export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

MODEL_KEY="qwen3_vl_30b"
PROMPT_PATH="prompts/scene_description_2D_parse.txt"

DATASET_DIR="data/probing/scene_description_balanced_2d"
OUT_DIR="data/probing/scene_description_balanced_2d_out_qwen3_vl_30b"

N_OBJECTS=10
N_TRIALS=100
IMG_SIZE=40
SEED=0

LAYERS="0,10,20"

echo "[$(date --iso-8601=seconds)] 1) Generate masked dataset"
python probing/00_generate_masked_scene_desc_dataset.py \
  --out_dir "$DATASET_DIR" \
  --n_objects "$N_OBJECTS" --n_trials "$N_TRIALS" --img_size "$IMG_SIZE" --seed "$SEED"

echo "[$(date --iso-8601=seconds)] 2) Debug token positions (sanity check image token slicing)"
python probing/debug_token_positions.py \
  --model "$MODEL_KEY" \
  --prompt_path "$PROMPT_PATH" \
  --dataset_dir "$DATASET_DIR" \
  --device auto --dtype auto

echo "[$(date --iso-8601=seconds)] 3) Extract per-layer image-token hidden states"
python probing/10_extract_image_tokens.py \
  --dataset_dir "$DATASET_DIR" \
  --out_dir "$OUT_DIR" \
  --model "$MODEL_KEY" \
  --prompt_path "$PROMPT_PATH" \
  --layers "$LAYERS" \
  --device auto --dtype auto

echo "[$(date --iso-8601=seconds)] 4) Pool per-object embeddings using masks"
python probing/20_build_object_embeddings.py \
  --dataset_dir "$DATASET_DIR" \
  --tokens_dir "$OUT_DIR/tokens" \
  --out_dir "$OUT_DIR"

echo "[$(date --iso-8601=seconds)] 5) Train linear probes (default split by sample_id)"
python probing/30_train_probes.py \
  --embeddings_path "$OUT_DIR/embeddings.npz" \
  --out_dir "$OUT_DIR/probes" \
  --test_split 0.2 --seed "$SEED" --C 1.0

echo "[$(date --iso-8601=seconds)] Done"

