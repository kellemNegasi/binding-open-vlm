#!/bin/bash
# SLURM job for running the 3D VLM tasks via run_vlm.py.
# This only includes tasks that have corresponding 3D prompts in prompts/*_3D.txt
# and 3D datasets generated under data/vlm/3D/<task_name>/metadata.csv.
#
# Prereq: generate the 3D datasets first (and metadata):
#   bash ./generate_3d_vlm_datasets.sh --task all
# Or per-task:
#   bash ./generate_3d_vlm_datasets.sh --task conjunctive_search
#
# Then run this job to produce:
#   output/vlm/3D/<task_name>/<model_name>.csv
# And aggregate:
#   python aggregate_vlm_3d.py --model-name <model_name>

#SBATCH -J vlm_3d_scene_description_job
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

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

# Set the model you want to run.
MODEL_NAME="${MODEL_NAME:-qwen2.5-VL-7B-Instruct}"

require_metadata() {
  local task_name="$1"
  local meta="$PROJECT_DIR/data/vlm/3D/$task_name/metadata.csv"
  if [[ ! -f "$meta" ]]; then
    echo "ERROR: Missing 3D metadata: $meta" >&2
    echo "Generate it first, e.g.: bash ./generate_3d_vlm_datasets.sh --task $task_name" >&2
    exit 1
  fi
}

COUNTING_TASKS=(
  counting_low_diversity
  counting_high_diversity
  counting_distinct
)

SEARCH_TASKS=(
  conjunctive_search
  disjunctive_search
)

SCENE_TASKS=(
  scene_description
)

ALL_TASKS=(
  "${COUNTING_TASKS[@]}"
  "${SEARCH_TASKS[@]}"
  "${SCENE_TASKS[@]}"
)

for TASK_NAME in "${ALL_TASKS[@]}"; do
  require_metadata "$TASK_NAME"
done

for TASK_NAME in "${ALL_TASKS[@]}"; do
  echo "[$(date --iso-8601=seconds)] Starting 3D task ${TASK_NAME} (model=${MODEL_NAME})"
  python run_vlm.py \
    "model=${MODEL_NAME}" \
    "task=${TASK_NAME}" \
    task.task_variant=3D \
    paths.root_dir="$PROJECT_DIR" \
    paths.data_dir="$PROJECT_DIR/data" \
    paths.output_dir="$PROJECT_DIR/output"
  echo "[$(date --iso-8601=seconds)] Finished 3D task ${TASK_NAME} (model=${MODEL_NAME})"
done

echo "[$(date --iso-8601=seconds)] Aggregating 3D results (model=${MODEL_NAME})"
python aggregate_vlm_3d.py --model-name "${MODEL_NAME}"
echo "[$(date --iso-8601=seconds)] Done."

