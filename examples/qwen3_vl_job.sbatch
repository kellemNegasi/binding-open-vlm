#!/bin/bash
# SLURM job for running examples/test_qwen3_vl.py on Rocket.
# Customize SBATCH directives below to match your quota and runtime needs.
#SBATCH -J qwen3_vl_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --mem=64G
#SBATCH -t 02:00:00


set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

# Point caches to $SCRATCH (faster + avoids filling home).
# SCRATCH_DIR="${SCRATCH:-$HOME}"
# export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
# export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
# export LMDEPLOY_CACHE="${LMDEPLOY_CACHE:-$SCRATCH/.cache/lmdeploy}"

# Optional: set your Hugging Face token if the model is gated
# export HUGGING_FACE_HUB_TOKEN="hf_xxx"

# python examples/test_qwen3_vl_transformers.py \
#   --image "$PROJECT_DIR/examples/sample-task.jpg" \
#   --output "$PROJECT_DIR/qwen3_job_output.txt"

# python examples/test_qwen3_vl.py \
#   --image "$PROJECT_DIR/examples/sample-task.jpg" \
#   --output "$PROJECT_DIR/qwen3_job_output_openvlm.txt"

# Run scene description with the local Qwen3-VL-30B model via Hydra.
export PROJECT_ROOT="$PROJECT_DIR"
python run_vlm.py \
  model=qwen3_vl_30b \
  task=scene_description \
  paths.root_dir="$PROJECT_DIR" \
  paths.data_dir="$PROJECT_DIR/data" \
  paths.output_dir="$PROJECT_DIR/output"
