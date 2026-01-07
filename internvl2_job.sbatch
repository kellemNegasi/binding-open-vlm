#!/bin/bash
#SBATCH -J internvl2_26b
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

PROJECT_DIR="$HOME/llama_vision/binding-open-vlm"
cd "$PROJECT_DIR"

export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

python run_vlm.py \
  model=internvl2_26b \
  task=scene_description \
  paths.root_dir="$PROJECT_DIR" \
  paths.data_dir="$PROJECT_DIR/data" \
  paths.output_dir="$PROJECT_DIR/output"
