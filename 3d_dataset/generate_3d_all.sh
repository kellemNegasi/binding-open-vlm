#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINDING_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
BLENDER_BIN="${BLENDER_BIN:-blender}"

NUM_SCENES="${NUM_SCENES:-1000}"
COUNT_MIN="${COUNT_MIN:-1}"
COUNT_MAX="${COUNT_MAX:-20}"
DISTRACTOR_MIN="${DISTRACTOR_MIN:-4}"
DISTRACTOR_MAX="${DISTRACTOR_MAX:-50}"

WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
RENDER_SAMPLES="${RENDER_SAMPLES:-256}"
MIN_PIXELS_PER_OBJECT="${MIN_PIXELS_PER_OBJECT:-10}"
MAX_LAYOUT_ATTEMPTS="${MAX_LAYOUT_ATTEMPTS:-20}"
MAX_RETRIES="${MAX_RETRIES:-50}"

DEVICE="${DEVICE:-auto}"
SAVE_BLENDFILES=0

SCENE_MIN_OBJECTS="${SCENE_MIN_OBJECTS:-8}"
SCENE_MAX_OBJECTS="${SCENE_MAX_OBJECTS:-12}"
SCENE_TRIPLET_TARGETS="${SCENE_TRIPLET_TARGETS:-0:200,1:200,2:200,3:200,4:200}"
SCENE_NUM_SCENES=""

ONLY_TASKS=""
PREVIEW=0
JOBS=1
JOB_IDX=""
NO_MERGE=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Generates the 3D datasets used in the binding-paper experiments:
  - disjunctive_search
  - conjunctive_search
  - counting_{low_entropy,medium_color,medium_shape,high_entropy}
  - scene_description

Options:
  --num-scenes N              Scenes for search + counting (default: ${NUM_SCENES})
  --preview                   Small render for visual inspection (sets 256x256, 32 samples)
  --jobs N                    Number of shards to split each task across (default: ${JOBS})
  --job-idx K                 Run only shard K (0-indexed) instead of launching all shards locally
  --no-merge                  Do not merge per-shard metadata into metadata.csv
  --width N                   Render width (default: ${WIDTH})
  --height N                  Render height (default: ${HEIGHT})
  --render-samples N          Cycles samples (default: ${RENDER_SAMPLES})
  --min-pixels-per-object N   Visibility threshold (default: ${MIN_PIXELS_PER_OBJECT})
  --max-layout-attempts N     Whole-scene layout retries (default: ${MAX_LAYOUT_ATTEMPTS})
  --max-retries N             Per-object placement retries inside Blender (default: ${MAX_RETRIES})
  --device <auto|cpu|cuda|hip|oneapi|metal>  Cycles backend (default: ${DEVICE})
  --blender-binary PATH       Blender binary (default: ${BLENDER_BIN})
  --save-blendfiles           Save per-scene .blendfiles

Scene-description options:
  --scene-num-scenes N        Override total scene_description count by distributing across triplets 0..4
  --scene-triplet-targets STR Explicit histogram "0:200,1:200,..." (default: ${SCENE_TRIPLET_TARGETS})

Task selection:
  --only TASKS                Comma-separated list of tasks to run

Environment overrides (alternative to flags):
  PYTHON_BIN, BLENDER_BIN, NUM_SCENES, WIDTH, HEIGHT, RENDER_SAMPLES, MIN_PIXELS_PER_OBJECT,
  MAX_LAYOUT_ATTEMPTS, MAX_RETRIES, DEVICE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-scenes) NUM_SCENES="$2"; shift 2;;
    --jobs) JOBS="$2"; shift 2;;
    --job-idx) JOB_IDX="$2"; shift 2;;
    --no-merge) NO_MERGE=1; shift 1;;
    --width) WIDTH="$2"; shift 2;;
    --height) HEIGHT="$2"; shift 2;;
    --render-samples) RENDER_SAMPLES="$2"; shift 2;;
    --min-pixels-per-object) MIN_PIXELS_PER_OBJECT="$2"; shift 2;;
    --max-layout-attempts) MAX_LAYOUT_ATTEMPTS="$2"; shift 2;;
    --max-retries) MAX_RETRIES="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --blender-binary) BLENDER_BIN="$2"; shift 2;;
    --save-blendfiles) SAVE_BLENDFILES=1; shift 1;;
    --scene-num-scenes) SCENE_NUM_SCENES="$2"; shift 2;;
    --scene-triplet-targets) SCENE_TRIPLET_TARGETS="$2"; shift 2;;
    --only) ONLY_TASKS="$2"; shift 2;;
    --preview) PREVIEW=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2;;
  esac
done

if [[ "${PREVIEW}" == "1" ]]; then
  WIDTH=256
  HEIGHT=256
  RENDER_SAMPLES=32
  MIN_PIXELS_PER_OBJECT=10
  MAX_LAYOUT_ATTEMPTS=10
fi

cd "${BINDING_ROOT}"

if [[ "${JOBS}" -lt 1 ]]; then
  echo "--jobs must be >= 1" >&2
  exit 2
fi

if [[ -n "${JOB_IDX}" ]]; then
  if [[ "${JOB_IDX}" -lt 0 || "${JOB_IDX}" -ge "${JOBS}" ]]; then
    echo "--job-idx must be in [0, --jobs)" >&2
    exit 2
  fi
fi

properties_json="${BINDING_ROOT}/3d_dataset/blender/data/properties.json"

shape_count="$("${PYTHON_BIN}" - <<PY
import json, pathlib
p = pathlib.Path(${properties_json@Q})
props = json.loads(p.read_text())
print(len(props.get("shapes", {})))
PY
)"

color_count="$("${PYTHON_BIN}" - <<PY
import json, pathlib
p = pathlib.Path(${properties_json@Q})
props = json.loads(p.read_text())
print(len(props.get("colors", {})))
PY
)"

cap() {
  local requested="$1"
  local limit="$2"
  if (( requested > limit )); then
    echo "${limit}"
  else
    echo "${requested}"
  fi
}

run_task() {
  local task_name="$1"
  shift

  echo "[generate_3d_all] task=${task_name}"
  "${PYTHON_BIN}" -m 3d_dataset.gen_blender \
    --task "${task_name}" \
    --root-dir "${BINDING_ROOT}" \
    --data-dir "${BINDING_ROOT}/data" \
    --device "${DEVICE}" \
    --blender-binary "${BLENDER_BIN}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --render-samples "${RENDER_SAMPLES}" \
    --min-pixels-per-object "${MIN_PIXELS_PER_OBJECT}" \
    --max-layout-attempts "${MAX_LAYOUT_ATTEMPTS}" \
    --max-retries "${MAX_RETRIES}" \
    "$@"
}

merge_metadata() {
  local task_name="$1"
  local task_dir="${BINDING_ROOT}/data/vlm/3D/${task_name}"
  local out="${task_dir}/metadata.csv"

  if [[ "${NO_MERGE}" == "1" ]]; then
    return 0
  fi

  # Merge with Python stdlib (no pandas dependency).
  "${PYTHON_BIN}" - <<PY
import csv, glob, os, sys
task_dir = ${task_dir@Q}
out_path = ${out@Q}
jobs = int(${JOBS})
parts = [os.path.join(task_dir, f"metadata.part-{i}.csv") for i in range(jobs)]
parts = [p for p in parts if os.path.exists(p)]
if not parts:
  print(f"[merge_metadata] No parts found for {task_dir}, skipping", file=sys.stderr)
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
      raise SystemExit(f"[merge_metadata] Header mismatch in {p}")
    for r in reader:
      rows.append(r)
if header is None:
  raise SystemExit(f"[merge_metadata] No readable parts for {task_dir}")
os.makedirs(task_dir, exist_ok=True)
with open(out_path, "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerow(header)
  writer.writerows(rows)
print(f"[merge_metadata] Wrote {out_path} from {len(parts)} shards ({len(rows)} rows)")
PY
}

run_task_sharded() {
  local task_name="$1"
  shift

  if [[ "${JOBS}" == "1" ]]; then
    run_task "${task_name}" --metadata-file metadata.csv --combined-scene-file scenes_combined.json "$@"
    return 0
  fi

  if [[ -n "${JOB_IDX}" ]]; then
    run_task "${task_name}" \
      --num-shards "${JOBS}" \
      --shard-index "${JOB_IDX}" \
      --metadata-file "metadata.part-${JOB_IDX}.csv" \
      --combined-scene-file "scenes_combined.part-${JOB_IDX}.json" \
      "$@"
    return 0
  fi

  echo "[generate_3d_all] sharding ${task_name} across ${JOBS} jobs"
  pids=()
  for (( shard=0; shard<"${JOBS}"; shard++ )); do
    run_task "${task_name}" \
      --num-shards "${JOBS}" \
      --shard-index "${shard}" \
      --metadata-file "metadata.part-${shard}.csv" \
      --combined-scene-file "scenes_combined.part-${shard}.json" \
      "$@" &
    pids+=("$!")
  done
  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      fail=1
    fi
  done
  if [[ "${fail}" == "1" ]]; then
    echo "[generate_3d_all] ERROR: one or more shards failed for ${task_name}" >&2
    exit 1
  fi
  merge_metadata "${task_name}"
}

should_run() {
  local task="$1"
  if [[ -z "${ONLY_TASKS}" ]]; then
    return 0
  fi
  IFS=',' read -r -a selected <<< "${ONLY_TASKS}"
  for t in "${selected[@]}"; do
    if [[ "${t}" == "${task}" ]]; then
      return 0
    fi
  done
  return 1
}

save_flag=()
if [[ "${SAVE_BLENDFILES}" == "1" ]]; then
  save_flag+=(--save-blendfiles)
fi

if should_run "disjunctive_search"; then
  run_task_sharded disjunctive_search \
    --num-scenes "${NUM_SCENES}" \
    --distractor-min "${DISTRACTOR_MIN}" \
    --distractor-max "${DISTRACTOR_MAX}" \
    "${save_flag[@]}"
fi

if should_run "conjunctive_search"; then
  run_task_sharded conjunctive_search \
    --num-scenes "${NUM_SCENES}" \
    --distractor-min "${DISTRACTOR_MIN}" \
    --distractor-max "${DISTRACTOR_MAX}" \
    "${save_flag[@]}"
fi

# Counting: cap requested count_max based on the current vocabulary so strict uniqueness modes don't error.
count_max_low="${COUNT_MAX}"
count_max_medium_shape="$(cap "${COUNT_MAX}" "${color_count}")"
count_max_medium_color="$(cap "${COUNT_MAX}" "${shape_count}")"
count_max_high="$(cap "${COUNT_MAX}" "$(cap "${shape_count}" "${color_count}")")"

if should_run "counting_low_entropy"; then
  run_task_sharded counting_low_entropy \
    --num-scenes "${NUM_SCENES}" \
    --count-min "${COUNT_MIN}" \
    --count-max "${count_max_low}" \
    "${save_flag[@]}"
fi

if should_run "counting_medium_shape"; then
  run_task_sharded counting_medium_shape \
    --num-scenes "${NUM_SCENES}" \
    --count-min "${COUNT_MIN}" \
    --count-max "${count_max_medium_shape}" \
    "${save_flag[@]}"
fi

if should_run "counting_medium_color"; then
  run_task_sharded counting_medium_color \
    --num-scenes "${NUM_SCENES}" \
    --count-min "${COUNT_MIN}" \
    --count-max "${count_max_medium_color}" \
    "${save_flag[@]}"
fi

if should_run "counting_high_entropy"; then
  run_task_sharded counting_high_entropy \
    --num-scenes "${NUM_SCENES}" \
    --count-min "${COUNT_MIN}" \
    --count-max "${count_max_high}" \
    "${save_flag[@]}"
fi

if should_run "scene_description"; then
  triplets="${SCENE_TRIPLET_TARGETS}"
  if [[ -n "${SCENE_NUM_SCENES}" ]]; then
    # Distribute requested count evenly over triplet bins 0..4 (matches default bins).
    triplets="$("${PYTHON_BIN}" - <<PY
total = int(${SCENE_NUM_SCENES})
bins = [0, 1, 2, 3, 4]
base = total // len(bins)
rem = total % len(bins)
parts = []
for i, b in enumerate(bins):
  parts.append(f"{b}:{base + (1 if i < rem else 0)}")
print(",".join(parts))
PY
)"
  fi

  run_task_sharded scene_description \
    --triplet-targets "${triplets}" \
    --scene-min-objects "${SCENE_MIN_OBJECTS}" \
    --scene-max-objects "${SCENE_MAX_OBJECTS}" \
    "${save_flag[@]}"
fi

echo "[generate_3d_all] done"
