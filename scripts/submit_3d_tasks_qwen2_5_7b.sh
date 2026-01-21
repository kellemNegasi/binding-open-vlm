#!/bin/bash
set -euo pipefail

SBATCH_SOURCE="${1:-experiments_job_3d.sbatch}"
MODEL_NAME="${MODEL_NAME:-qwen2.5-VL-7B-Instruct}"

if [[ ! -f "$SBATCH_SOURCE" ]]; then
  echo "ERROR: Missing SBATCH source: $SBATCH_SOURCE" >&2
  exit 1
fi

readarray -t SBATCH_LINES < <(awk '/^#SBATCH /{print}' "$SBATCH_SOURCE")
if [[ ${#SBATCH_LINES[@]} -eq 0 ]]; then
  echo "ERROR: No #SBATCH directives found in $SBATCH_SOURCE" >&2
  exit 1
fi

# Tasks copied from experiments_job_3d.sbatch (all available 3D tasks).
TASKS=(
  counting_low_diversity
  counting_high_diversity
  counting_distinct
  conjunctive_search
  disjunctive_search
  scene_description
)

if [[ ${#TASKS[@]} -eq 0 ]]; then
  echo "ERROR: No tasks found in $SBATCH_SOURCE" >&2
  exit 1
fi

SBATCH_BLOCK="$(printf "%s\n" "${SBATCH_LINES[@]}")"

for TASK_NAME in "${TASKS[@]}"; do
  sbatch <<EOF
#!/bin/bash
${SBATCH_BLOCK}

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="\$HOME/binding-open-vlm"
cd "\$PROJECT_DIR"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="\$PROJECT_DIR"

MODEL_NAME="${MODEL_NAME}"
TASK_NAME="${TASK_NAME}"

require_metadata() {
  local task_name="\$1"
  local meta="\$PROJECT_DIR/data/vlm/3D/\$task_name/metadata.csv"
  if [[ ! -f "\$meta" ]]; then
    echo "ERROR: Missing 3D metadata: \$meta" >&2
    echo "Generate it first, e.g.: bash ./generate_3d_vlm_datasets.sh --task \$task_name" >&2
    exit 1
  fi
}

require_metadata "\$TASK_NAME"

echo "[\$(date --iso-8601=seconds)] Starting 3D task \${TASK_NAME} (model=\${MODEL_NAME})"
python run_vlm.py \
  "model=\${MODEL_NAME}" \
  "task=\${TASK_NAME}" \
  task.task_variant=3D \
  paths.root_dir="\$PROJECT_DIR" \
  paths.data_dir="\$PROJECT_DIR/data" \
  paths.output_dir="\$PROJECT_DIR/output"
echo "[\$(date --iso-8601=seconds)] Finished 3D task \${TASK_NAME} (model=\${MODEL_NAME})"

echo "[\$(date --iso-8601=seconds)] Aggregating 3D results (model=\${MODEL_NAME})"
python aggregate_vlm_3d.py --model-name "\${MODEL_NAME}"
echo "[\$(date --iso-8601=seconds)] Done."
EOF
done
