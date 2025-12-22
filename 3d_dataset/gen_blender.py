from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from .devices import cycles_device_flag, detect_device
from .sampling import (
    SceneBlueprint,
    SceneVocabulary,
    make_conjunctive_scenes,
    make_counting_scenes,
    make_disjunctive_scenes,
    make_scene_description_scenes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(__file__).resolve().parent / "blender"
PROPERTIES_PATH = ASSET_ROOT / "data" / "properties.json"
RENDER_SCRIPT = ASSET_ROOT / "render_binding.py"


def log(msg: str) -> None:
    print(f"[gen_blender] {msg}", flush=True)


def load_properties(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_triplet_targets(arg: str) -> Dict[int, int]:
    """
    Parse strings like "0:200,1:200,2:200" into {0: 200, 1: 200, 2: 200}.
    """
    targets: Dict[int, int] = {}
    pairs = [entry.strip() for entry in arg.split(",") if entry.strip()]
    for pair in pairs:
        triplet_str, count_str = pair.split(":")
        targets[int(triplet_str)] = int(count_str)
    if not targets:
        raise ValueError("At least one triplet target must be supplied.")
    return targets


def prepare_scene_specs(
    *,
    task_name: str,
    blueprints: List[SceneBlueprint],
    data_dir: Path,
    root_dir: Path,
    save_blendfiles: bool,
) -> Tuple[List[Dict], pd.DataFrame]:
    task_root = data_dir / "vlm" / "3D" / task_name
    image_dir = task_root / "images"
    scene_dir = task_root / "scenes"
    blend_dir = task_root / "blendfiles"
    for folder in (image_dir, scene_dir):
        folder.mkdir(parents=True, exist_ok=True)
    if save_blendfiles:
        blend_dir.mkdir(parents=True, exist_ok=True)

    scene_specs: List[Dict] = []
    metadata_rows: List[Dict] = []
    for idx, blueprint in enumerate(blueprints):
        fname = f"{idx:06d}_{blueprint.slug}"
        image_path = image_dir / f"{fname}.png"
        scene_path = scene_dir / f"{fname}.json"
        blend_path = str(blend_dir / f"{fname}.blend") if save_blendfiles else None
        rel_path = os.path.relpath(image_path, root_dir)
        scene_specs.append(
            {
                "scene_id": f"{task_name}_{idx:06d}",
                "task_name": task_name,
                "split": "3D",
                "output_image": str(image_path),
                "output_scene": str(scene_path),
                "output_blendfile": blend_path,
                "objects": [obj.to_dict() for obj in blueprint.objects],
                "metadata": blueprint.metadata,
            }
        )
        row = {"path": rel_path}
        row.update(blueprint.metadata)
        if "response" not in row:
            row["response"] = pd.NA
        if "answer" not in row:
            row["answer"] = pd.NA
        metadata_rows.append(row)
    return scene_specs, pd.DataFrame(metadata_rows)


def run_blender(
    *,
    scene_specs: List[Dict],
    device_choice: str,
    width: int,
    height: int,
    render_samples: int,
    render_tile_size: int,
    min_pixels_per_object: int,
    max_layout_attempts: int,
    output_scene_file: Path,
    blender_binary: str,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump({"scenes": scene_specs}, tmp)
        tmp_path = tmp.name

    compute_flag = cycles_device_flag(device_choice)
    log(
        f"Launching Blender with {len(scene_specs)} scenes | device='{device_choice}' "
        f"(Cycles flag={compute_flag})"
    )

    cmd = [
        blender_binary,
        "--background",
        "-noaudio",
        "--python",
        str(RENDER_SCRIPT),
        "--",
        f"--scene_specs_path={tmp_path}",
        f"--base_scene_blendfile={ASSET_ROOT / 'data/base_scene.blend'}",
        f"--properties_json={PROPERTIES_PATH}",
        f"--shape_dir={ASSET_ROOT / 'data/shapes'}",
        f"--material_dir={ASSET_ROOT / 'data/materials'}",
        f"--compute_device_type={compute_flag}",
        f"--width={width}",
        f"--height={height}",
        f"--render_num_samples={render_samples}",
        f"--render_tile_size={render_tile_size}",
        f"--min_pixels_per_object={min_pixels_per_object}",
        f"--max_layout_attempts={max_layout_attempts}",
        f"--output_scene_file={output_scene_file}",
        f"--num_images={len(scene_specs)}",
    ]
    try:
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D datasets for binding tasks.")
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "disjunctive_search",
            "conjunctive_search",
            "counting_low_entropy",
            "counting_medium_color",
            "counting_medium_shape",
            "counting_high_entropy",
            "scene_description",
        ],
    )
    parser.add_argument("--num-scenes", type=int, default=1000, help="Number of scenes to render.")
    parser.add_argument("--distractor-min", type=int, default=4)
    parser.add_argument("--distractor-max", type=int, default=50)
    parser.add_argument("--count-min", type=int, default=1)
    parser.add_argument("--count-max", type=int, default=20)
    parser.add_argument(
        "--triplet-targets",
        type=str,
        default="0:200,1:200,2:200,3:200,4:200",
        help="Scene-description histogram as 'triplets:count,...'.",
    )
    parser.add_argument("--scene-min-objects", type=int, default=8)
    parser.add_argument("--scene-max-objects", type=int, default=12)
    parser.add_argument("--device", default="auto", help="Compute device preference (auto/cpu/gpu/xpu/etc).")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--root-dir", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--metadata-file", default="metadata.csv")
    parser.add_argument("--save-blendfiles", action="store_true")
    parser.add_argument("--blender-binary", default="blender")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--render-samples", type=int, default=256)
    parser.add_argument(
        "--render-tile-size",
        "--render_tile_size",
        dest="render_tile_size",
        default=256,
        type=int,
        help="Tile size (kept for compatibility; may be ignored in newer Blender).",
    )
    parser.add_argument(
        "--min-pixels-per-object",
        dest="min_pixels_per_object",
        type=int,
        default=10,
        help="Minimum visible pixels per object before a layout retry (forwarded to Blender).",
    )
    parser.add_argument(
        "--max-layout-attempts",
        dest="max_layout_attempts",
        type=int,
        default=20,
        help="Maximum number of whole-scene layout retries before proceeding anyway.",
    )

    return parser.parse_args(argv)


def build_blueprints(args: argparse.Namespace, vocab: SceneVocabulary) -> List[SceneBlueprint]:
    rng = random.Random(args.seed)
    if args.task == "disjunctive_search":
        return make_disjunctive_scenes(
            num_images=args.num_scenes,
            distractor_range=range(args.distractor_min, args.distractor_max + 1),
            vocab=vocab,
            rng=rng,
            object_size="small",
        )
    if args.task == "conjunctive_search":
        return make_conjunctive_scenes(
            num_images=args.num_scenes,
            distractor_range=range(args.distractor_min, args.distractor_max + 1),
            vocab=vocab,
            rng=rng,
            object_size="small",
        )
    if args.task.startswith("counting_"):
        mode = args.task.replace("counting_", "")
        return make_counting_scenes(
            num_images=args.num_scenes,
            count_range=range(args.count_min, args.count_max + 1),
            mode=mode,
            vocab=vocab,
            rng=rng,
            object_size="small",
        )
    if args.task == "scene_description":
        triplet_targets = parse_triplet_targets(args.triplet_targets)
        return make_scene_description_scenes(
            triplet_targets=triplet_targets,
            object_range=(args.scene_min_objects, args.scene_max_objects),
            vocab=vocab,
            rng=rng,
            object_size="small",
        )
    raise ValueError(f"Unrecognized task {args.task}")


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if not RENDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing Blender render script at {RENDER_SCRIPT}")
    properties = load_properties(PROPERTIES_PATH)
    vocab = SceneVocabulary.from_properties(properties)
    blueprints = build_blueprints(args, vocab)
    if args.task == "scene_description" and args.num_scenes not in (0, len(blueprints)):
        print(
            f"[warning] Ignoring --num-scenes because the triplet histogram expands to {len(blueprints)} scenes.",
            file=sys.stderr,
        )
    scene_specs, metadata_df = prepare_scene_specs(
        task_name=args.task,
        blueprints=blueprints,
        data_dir=args.data_dir,
        root_dir=args.root_dir,
        save_blendfiles=args.save_blendfiles,
    )
    metadata_path = args.data_dir / "vlm" / "3D" / args.task / args.metadata_file
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(metadata_path, index=False)
    device_choice = detect_device(args.device)
    log(f"Device preference '{args.device}' resolved to '{device_choice}'")
    output_scene_file = metadata_path.parent / "scenes_combined.json"
    output_scene_file.parent.mkdir(parents=True, exist_ok=True)
    run_blender(
        scene_specs=scene_specs,
        device_choice=device_choice,
        width=args.width,
        height=args.height,
        render_samples=args.render_samples,
        render_tile_size=args.render_tile_size,
        min_pixels_per_object=args.min_pixels_per_object,
        max_layout_attempts=args.max_layout_attempts,
        output_scene_file=output_scene_file,
        blender_binary=args.blender_binary,
    )
    print(f"Wrote metadata to {metadata_path}")
    print(f"Rendered {len(scene_specs)} scenes for {args.task}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
