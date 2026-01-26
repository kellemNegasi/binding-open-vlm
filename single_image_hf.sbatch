#!/bin/bash
#SBATCH -J qwen25_vl_single_image_hf
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --mem=64G
#SBATCH -t 01:00:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

source ~/binding-open-vlm/.venv/bin/activate

PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

SEARCH_IMG="data/vlm/3D/disjunctive_search/images/disjunctive_search_uniform_n35_003508.png"
COUNT_IMG="data/vlm/3D/counting_high_diversity/images/counting_high_diversity_n10_001000.png"

SEARCH_PROMPT="$(cat prompts/disjunctive_search_3D.txt)"
COUNT_PROMPT="$(cat prompts/counting_3D.txt)"

python examples/test_qwen3_vl_transformers.py \
  --weights "model-weights/Qwen2.5-VL-32B-Instruct-AWQ" \
  --image "$SEARCH_IMG" \
  --prompt "$SEARCH_PROMPT" \
  --output "qwen25_search_hf.txt" \
  --dtype float16 \
  --device-map auto \
  --max-new-tokens 512

python examples/test_qwen3_vl_transformers.py \
  --weights "model-weights/Qwen2.5-VL-32B-Instruct-AWQ" \
  --image "$COUNT_IMG" \
  --prompt "$COUNT_PROMPT" \
  --output "qwen25_count_hf.txt" \
  --dtype float16 \
  --device-map auto \
  --max-new-tokens 512
