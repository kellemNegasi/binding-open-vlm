#!/bin/bash
#SBATCH -J idefics2_vlm_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate virtual environment
source ~/binding-open-vlm/.venv/bin/activate

# Project root
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

# Hugging Face + lmdeploy cache (recommended)
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Optional (only if model access is gated)
# export HUGGING_FACE_HUB_TOKEN="hf_xxx"

# Run IDEFICS2 test script
python examples/test_idefics2_vl.py \
  --image "$PROJECT_DIR/examples/sample-task.jpg" \
  --output "$PROJECT_DIR/idefics2_job_output.txt"
