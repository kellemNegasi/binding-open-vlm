#!/bin/bash
#SBATCH -J internvl2_26b_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda/12.1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm

PROJECT_DIR="$HOME/llama_vision/binding-open-vlm"
cd "$PROJECT_DIR"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"
export HF_HOME="${SCRATCH:-$HOME}/.cache/huggingface"

COUNTING_TASKS=(
  counting_control_shape
  counting_control
  counting_distinct
  counting_high_diversity
  counting_low_diversity
)

for TASK_NAME in "${COUNTING_TASKS[@]}"; do
  python run_vlm.py \
    model=internvl2_26b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
done

SEARCH_TASKS=(
  conjunctive_search
  disjunctive_search
  disjunctive_search_control
)

for TASK_NAME in "${SEARCH_TASKS[@]}"; do
  python run_vlm.py \
    model=internvl2_26b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
done

SCENE_TASKS=(
  scene_description
  scene_description_BALANCED
)

for TASK_NAME in "${SCENE_TASKS[@]}"; do
  python run_vlm.py \
    model=internvl2_26b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
done

RMTS_CONDITIONS=(
  unified
  decomposed
)

RMTS_SUBTASKS=(
  features
  features2
  relations
  full
)

for CONDITION in "${RMTS_CONDITIONS[@]}"; do
  for SUBTASK in "${RMTS_SUBTASKS[@]}"; do
    python run_vlm.py \
      model=internvl2_26b \
      task=rmts \
      task.condition="${CONDITION}" \
      task.subtask="${SUBTASK}" \
      paths.root_dir="$PROJECT_DIR" \
      paths.data_dir="$PROJECT_DIR/data" \
      paths.output_dir="$PROJECT_DIR/output"
  done
done
