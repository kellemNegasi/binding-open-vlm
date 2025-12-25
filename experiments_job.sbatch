#!/bin/bash
# SLURM job for running Qwen2.5-VL tasks via run_vlm.py on Rocket.
# Customize SBATCH directives below to match your quota and runtime needs.
#SBATCH -J qwen2_5_vl_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
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

# Run the all tasks sequentially via Hydra.
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"
COUNTING_TASKS=(
  counting_control_shape
  counting_control
  counting_distinct
  counting_high_diversity
  counting_low_diversity
)

for TASK_NAME in "${COUNTING_TASKS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Starting task ${TASK_NAME}"
  python run_vlm.py \
    model=qwen3_vl_30b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
  echo "[$(date --iso-8601=seconds)] Finished task ${TASK_NAME}"
done

SEARCH_TASKS=(
  conjunctive_search
  disjunctive_search
  disjunctive_search_control
)

for TASK_NAME in "${SEARCH_TASKS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Starting task ${TASK_NAME}"
  python run_vlm.py \
    model=qwen3_vl_30b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
  echo "[$(date --iso-8601=seconds)] Finished task ${TASK_NAME}"
done

SCENE_TASKS=(
  scene_description_BALANCED
)

for TASK_NAME in "${SCENE_TASKS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Starting task ${TASK_NAME}"
  python run_vlm.py \
    model=qwen3_vl_30b \
    task="${TASK_NAME}" \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
  echo "[$(date --iso-8601=seconds)] Finished task ${TASK_NAME}"
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
    echo "[$(date --iso-8601=seconds)] Starting RMTS condition=${CONDITION} subtask=${SUBTASK}"
    python run_vlm.py \
      model=qwen3_vl_30b \
      task=rmts \
      task.condition="${CONDITION}" \
      task.subtask="${SUBTASK}" \
      paths.root_dir="$PROJECT_DIR" \
      paths.data_dir="$PROJECT_DIR/data" \
      paths.output_dir="$PROJECT_DIR/output"
    echo "[$(date --iso-8601=seconds)] Finished RMTS condition=${CONDITION} subtask=${SUBTASK}"
  done
done
