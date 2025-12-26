#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") --jobs N [task...]

Merge per-shard metadata files (metadata.part-K.csv) into metadata.csv.

Examples:
  ./3d_dataset/merge_3d_metadata.sh --jobs 8 disjunctive_search
  ./3d_dataset/merge_3d_metadata.sh --jobs 4 disjunctive_search conjunctive_search counting_high_entropy
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

if [[ "$1" != "--jobs" ]]; then
  usage
  exit 2
fi
JOBS="$2"
shift 2

if [[ "${JOBS}" -lt 1 ]]; then
  echo "--jobs must be >= 1" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  # Default: merge any task directories that contain part files.
  mapfile -t tasks < <(find "${BINDING_ROOT}/data/vlm/3D" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
else
  tasks=("$@")
fi

for task in "${tasks[@]}"; do
  task_dir="${BINDING_ROOT}/data/vlm/3D/${task}"
  out="${task_dir}/metadata.csv"

  "${PYTHON_BIN}" - <<PY
import csv, os, sys
task_dir = ${task_dir@Q}
out_path = ${out@Q}
jobs = int(${JOBS})
parts = [os.path.join(task_dir, f"metadata.part-{i}.csv") for i in range(jobs)]
parts = [p for p in parts if os.path.exists(p)]
if not parts:
  print(f"[merge_3d_metadata] {task_dir}: no parts found; skipping")
  sys.exit(0)
rows = []
header = None
for p in parts:
  with open(p, newline="") as f:
    reader = csv.reader(f)
    part_header = next(reader, None)
    if part_header is None:
      continue
    if header is None:
      header = part_header
    elif part_header != header:
      raise SystemExit(f"[merge_3d_metadata] header mismatch in {p}")
    rows.extend(reader)
if header is None:
  raise SystemExit(f"[merge_3d_metadata] no readable parts for {task_dir}")
os.makedirs(task_dir, exist_ok=True)
with open(out_path, "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(header)
  writer.writerows(rows)
print(f"[merge_3d_metadata] {task_dir}: wrote metadata.csv ({len(rows)} rows from {len(parts)} shards)")
PY
done
