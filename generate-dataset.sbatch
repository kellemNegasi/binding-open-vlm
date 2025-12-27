#!/bin/bash
# SLURM job for generating the 3D VLM datasets (runs generate_3d_vlm_datasets.sh).
#SBATCH -J scene_description
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=main
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

# Your Blender binary (no cluster install needed).
export BLENDER_BIN="$HOME/binary/blender-5.0.1-linux-x64/blender"
test -x "$BLENDER_BIN" || { echo "ERROR: BLENDER_BIN not executable: $BLENDER_BIN"; exit 1; }

# Where to write the generated dataset (default is $PROJECT_DIR/data)
# export DATA_DIR="$PROJECT_DIR/data"
# Or to scratch:
# export DATA_DIR="${SCRATCH:-$HOME}/binding-open-vlm-data"

# Optional generation knobs (override as needed)
export RENDER_SAMPLES="${RENDER_SAMPLES:-64}"
export N_TRIALS_COUNTING="${N_TRIALS_COUNTING:-100}"
export N_TRIALS_SEARCH="${N_TRIALS_SEARCH:-100}"
export N_TRIALS_SCENE="${N_TRIALS_SCENE:-100}"
export WIDTH="${WIDTH:-640}"
export HEIGHT="${HEIGHT:-480}"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

# Run the 3D dataset generation script
# Valid tasks are: all, conjunctive_search, disjunctive_search, counting_low_diversity, 
# counting_high_diversity, counting_distinct, scene_description
bash generate_3d_vlm_datasets.sh --task scene_description
