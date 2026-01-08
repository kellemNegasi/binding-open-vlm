#!/bin/bash
#SBATCH -J idefics2_vl_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

source ~/binding-open-vlm/.venv/bin/activate

PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

# Put HF cache on scratch (recommended on Rocket)
SCRATCH_DIR="${SCRATCH:-$HOME}"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"   # ok, even if deprecated warning
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

MODEL_NAME=idefics2_8b

COUNTING_TASKS=(
  counting_control_shape
  counting_control
  counting_distinct
  counting_high_diversity
  counting_low_diversity
)

SEARCH_TASKS=(
  conjunctive_search
  disjunctive_search
  disjunctive_search_control
)

SCENE_TASKS=(
  # scene_description
  scene_description_BALANCED 
)

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

run_task () {
  local TASK_NAME="$1"
  echo "[$(date --iso-8601=seconds)] Starting task ${TASK_NAME}"
  python run_vlm.py \
    model="$MODEL_NAME" \
    task="$TASK_NAME" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
  echo "[$(date --iso-8601=seconds)] Finished task ${TASK_NAME}"
}

for TASK_NAME in "${COUNTING_TASKS[@]}"; do
  run_task "$TASK_NAME"
done

# for TASK_NAME in "${SEARCH_TASKS[@]}"; do
#   run_task "$TASK_NAME"
# done

for TASK_NAME in "${SCENE_TASKS[@]}"; do
  run_task "$TASK_NAME"
done

# for CONDITION in "${RMTS_CONDITIONS[@]}"; do
#   for SUBTASK in "${RMTS_SUBTASKS[@]}"; do
#     echo "[$(date --iso-8601=seconds)] Starting RMTS condition=${CONDITION} subtask=${SUBTASK}"
#     python run_vlm.py \
#       model="$MODEL_NAME" \
#       task=rmts \
#       task.condition="$CONDITION" \
#       task.subtask="$SUBTASK" \
#       paths.root_dir="$PROJECT_DIR" \
#       paths.data_dir="$PROJECT_DIR/data" \
#       paths.output_dir="$PROJECT_DIR/output"
#     echo "[$(date --iso-8601=seconds)] Finished RMTS condition=${CONDITION} subtask=${SUBTASK}"
#   done
# done
