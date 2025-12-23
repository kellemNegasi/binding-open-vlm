#!/bin/bash
# SLURM job for running Qwen2-VL-7B on Rocket.
#SBATCH -J qwen2_vl_7b_job
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --mem=64G
#SBATCH -t 06:00:00

set -euo pipefail

# Caricamento moduli (come da tuo originale)
module load cuda/12.1
module load python/3.12.3

# Attivazione ambiente virtuale
source ~/binding-open-vlm/.venv/bin/activate

# Percorsi del progetto
PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

# Variabili d'ambiente
export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

# Esecuzione del task: nota model=qwen2_vl_7b che richiama il file .yaml creato prima
python run_vlm.py \
  model=qwen2_vl_7b \
  task=scene_description_BALANCED \
  paths.root_dir="$PROJECT_DIR" \
  paths.data_dir="$PROJECT_DIR/data" \
  paths.output_dir="$PROJECT_DIR/output"
