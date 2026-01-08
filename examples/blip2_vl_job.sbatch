#!/bin/bash
#SBATCH -J blip2_vlm_job
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

# Activate your virtual environment
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

# Hugging Face cache (optional but recommended)
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Optional: set your Hugging Face token (for gated models)
# export HUGGING_FACE_HUB_TOKEN="hf_..."

# Run BLIP-2 test script
python examples/test_blip2_vl.py \
  --image "$PROJECT_DIR/examples/sample-task.jpg" \
  --output "$PROJECT_DIR/blip2_job_output.txt"
