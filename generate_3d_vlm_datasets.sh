cat > generate_3d_vlm_datasets.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
BLENDER_BIN="${BLENDER_BIN:-blender}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

N_TRIALS_COUNTING="${N_TRIALS_COUNTING:-100}"
N_TRIALS_SEARCH="${N_TRIALS_SEARCH:-100}"
N_TRIALS_SCENE="${N_TRIALS_SCENE:-100}"

RENDER_SAMPLES="${RENDER_SAMPLES:-64}"   # lower = faster, higher = nicer
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

render() {
  local blender_task="$1"     # gen_blender.py --task (search/popout/counting_*/binding)
  local dataset_name="$2"     # must match Hydra task_name (folder name)
  local split="$3"            # used in filenames; we parse this for labels
  local n_objects="$4"
  local n_images="$5"
  shift 5
  local extra_args=("$@")

  local out="${DATA_DIR}/vlm/3D/${dataset_name}"
  mkdir -p "${out}/images" "${out}/scenes"

  "${BLENDER_BIN}" --background --python "${ROOT_DIR}/3d-dataset-generation/gen_blender.py" -- \
    --task "${blender_task}" \
    --num_objects "${n_objects}" \
    --num_images "${n_images}" \
    --split "${split}" \
    --filename_prefix "${dataset_name}" \
    --render_num_samples "${RENDER_SAMPLES}" \
    --width "${WIDTH}" --height "${HEIGHT}" \
    --base_scene_blendfile "${ROOT_DIR}/3d-dataset-generation/blender_utils/base_scene.blend" \
    --properties_json "${ROOT_DIR}/3d-dataset-generation/blender_utils/properties.json" \
    --shape_dir "${ROOT_DIR}/3d-dataset-generation/blender_utils/shapes" \
    --material_dir "${ROOT_DIR}/3d-dataset-generation/blender_utils/materials" \
    --output_image_dir "${out}/images" \
    --output_scene_dir "${out}/scenes" \
    --output_scene_file "${out}/scenes/${dataset_name}_${split}.json" \
    "${extra_args[@]}"
}

# 1) Conjunctive search (odd-ball present vs absent)
for n in 5 10 15 20 25 30 35 40 45 50; do
  render search                conjunctive_search "present_n${n}" "${n}" "${N_TRIALS_SEARCH}"
  render search_counterfactual conjunctive_search "absent_n${n}"  "${n}" "${N_TRIALS_SEARCH}"
done

# 2) Disjunctive search (popout present vs uniform)
for n in 5 10 15 20 25 30 35 40 45 50; do
  render popout                disjunctive_search "popout_n${n}"  "${n}" "${N_TRIALS_SEARCH}"
  render popout_counterfactual disjunctive_search "uniform_n${n}" "${n}" "${N_TRIALS_SEARCH}"
done

# 3) Counting variants (set sizes 1..20)
for n in $(seq 1 20); do
  render counting_min_distinctiveness counting_low_diversity  "n${n}" "${n}" "${N_TRIALS_COUNTING}"
  render counting_max_distinctiveness counting_high_diversity "n${n}" "${n}" "${N_TRIALS_COUNTING}"
  render counting_max_distinctiveness counting_distinct       "n${n}" "${n}" "${N_TRIALS_COUNTING}"
done

# 4) Scene description (uses binding renderer to get mixed shapes/colors)
for n in 10 11 12 13 14 15; do
  render binding scene_description "n${n}" "${n}" "${N_TRIALS_SCENE}" --num_features "${n}"
done

# 5) Build metadata.csv from per-image scene JSONs (what tasks/task.py expects)
ROOT_DIR="${ROOT_DIR}" DATA_DIR="${DATA_DIR}" "${PYTHON_BIN}" - <<'PY'
import csv, json, os, re
from pathlib import Path

root = Path(os.environ["ROOT_DIR"]).resolve()
data_dir = Path(os.environ["DATA_DIR"]).resolve()

datasets = {
  "conjunctive_search": "conjunctive",
  "disjunctive_search": "disjunctive",
  "counting_low_diversity": "counting",
  "counting_high_diversity": "counting",
  "counting_distinct": "counting",
  "scene_description": "scene_description",
}

def write_csv(path: Path, fieldnames, rows):
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
      w.writerow(r)

for name, kind in datasets.items():
  base = data_dir / "vlm" / "3D" / name
  scenes_dir = base / "scenes"
  images_dir = base / "images"
  if not scenes_dir.exists():
    continue

  pattern = re.compile(rf"^{re.escape(name)}_(?P<split>.+)_(?P<idx>\d{{6}})\.json$")
  rows = []

  for scene_path in sorted(scenes_dir.glob("*.json")):
    m = pattern.match(scene_path.name)
    if not m:
      continue
    split = m.group("split")
    scene = json.loads(scene_path.read_text())
    objects = scene.get("objects", []) or []
    img_path = images_dir / (scene_path.stem + ".png")
    if not img_path.exists():
      continue

    rel_path = os.path.relpath(str(img_path), str(root))
    n_objects = len(objects)

    if kind == "counting":
      rows.append({"path": rel_path, "n_objects": n_objects, "response": ""})
    elif kind == "conjunctive":
      incongruent = split.startswith("present_")
      rows.append({"path": rel_path, "n_objects": n_objects, "incongruent": incongruent, "response": "", "answer": ""})
    elif kind == "disjunctive":
      popout = split.startswith("popout_")
      rows.append({"path": rel_path, "n_objects": n_objects, "popout": popout, "response": "", "answer": ""})
    elif kind == "scene_description":
      feats = [{"shape": o.get("shape"), "color": o.get("color")} for o in objects]
      n_shapes = len({f["shape"] for f in feats if f.get("shape")})
      n_colors = len({f["color"] for f in feats if f.get("color")})
      rows.append({
        "path": rel_path,
        "n_objects": n_objects,
        "n_shapes": n_shapes,
        "n_colors": n_colors,
        "features": json.dumps(feats),
        "response": "",
        "answer": "",
      })
    else:
      raise RuntimeError(kind)

  if not rows:
    continue

  if kind == "counting":
    fields = ["path", "n_objects", "response"]
  elif kind == "conjunctive":
    fields = ["path", "n_objects", "incongruent", "response", "answer"]
  elif kind == "disjunctive":
    fields = ["path", "n_objects", "popout", "response", "answer"]
  else:
    fields = ["path", "n_objects", "n_shapes", "n_colors", "features", "response", "answer"]

  write_csv(base / "metadata.csv", fields, rows)
  print(f"Wrote {base/'metadata.csv'} ({len(rows)} rows)")
PY

echo "Done. Datasets are in: ${DATA_DIR}/vlm/3D/*"
SH

chmod +x generate_3d_vlm_datasets.sh

# run (set BLENDER_BIN if blender isn't on PATH)
BLENDER_BIN=blender ./generate_3d_vlm_datasets.sh
