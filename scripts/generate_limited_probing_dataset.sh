#!/usr/bin/env bash
set -euo pipefail

# Purpose: Generate a limited-vocab balanced scene description probing dataset
# (4 colors x 4 shapes) for reducing label dimensionality in global probing.

python probing/generate_masked_scene_desc_dataset.py \
  --config_task_yaml config/task/scene_description_BALANCED_limited.yaml \
  --out_dir data/probing/scene_description_balanced_2d_limited
