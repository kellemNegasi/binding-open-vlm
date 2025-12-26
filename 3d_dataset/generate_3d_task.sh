#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") <task> [options...]

Runs a single 3D generation task via the unified driver script.

Examples:
  ./3d_dataset/generate_3d_task.sh disjunctive_search --num-scenes 20 --preview
  ./3d_dataset/generate_3d_task.sh counting_high_entropy --num-scenes 200 --width 256 --height 256
  ./3d_dataset/generate_3d_task.sh scene_description --scene-num-scenes 50 --preview
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

task="$1"
shift

exec "${SCRIPT_DIR}/generate_3d_all.sh" --only "${task}" "$@"
