#!/bin/bash
#SBATCH -J qwen25_vl_single_image_lmdeploy
#SBATCH --output=slurm-%x.%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80g:2
#SBATCH --mem=32G
#SBATCH -t 00:30:00

set -euo pipefail

module load cuda/12.1
module load python/3.12.3

source ~/binding-open-vlm/.venv/bin/activate

PROJECT_DIR="$HOME/binding-open-vlm"
cd "$PROJECT_DIR"

export HYDRA_FULL_ERROR=1
export PROJECT_ROOT="$PROJECT_DIR"

SEARCH_IMG="data/vlm/3D/disjunctive_search/images/disjunctive_search_uniform_n35_003508.png"
COUNT_IMG="data/vlm/3D/counting_high_diversity/images/counting_high_diversity_n10_001000.png"

python - <<'PY'
from pathlib import Path
from PIL import Image
from lmdeploy import pipeline

weights = "model-weights/Qwen2.5-VL-32B-Instruct-AWQ"
pipe = pipeline(model_path=weights)

def wrap_qwen(prompt):
    return (
        "<|im_start|>user\n"
        "<image>\n"
        f"{prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def run(prompt, image_path):
    prompt = wrap_qwen(prompt)
    res = pipe([(prompt, Image.open(image_path))])[0]
    text = res.text if hasattr(res, "text") else str(res)
    print("PROMPT:", prompt[:80], "...")
    print("IMAGE:", image_path)
    print("OUT:", text[:200], "...")
    print("LEN:", len(text), "UNIQUE:", sorted(set(text))[:5])
    print("----")

run(
  Path("prompts/disjunctive_search_3D.txt").read_text(),
  "data/vlm/3D/disjunctive_search/images/disjunctive_search_uniform_n35_003508.png"
)
run(
  Path("prompts/counting_3D.txt").read_text(),
  "data/vlm/3D/counting_high_diversity/images/counting_high_diversity_n10_001000.png"
)
PY
