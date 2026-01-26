#!/bin/bash
# SLURM job for generating the limited-vocab probing dataset.
# Runs scripts/generate_limited_probing_dataset.sh on the cluster.
#SBATCH -J probing_limited_dataset
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=main
#SBATCH --mem=16G
#SBATCH -t 01:00:00

set -euo pipefail

module load python/3.12.3

# Activate your virtual environment. Update the path if needed.
source ~/binding-open-vlm/.venv/bin/activate

# Change this to the absolute path of the repo on the cluster.
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

bash scripts/generate_limited_probing_dataset.sh
