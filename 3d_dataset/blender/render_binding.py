# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""
Render binding-problem stimuli using Blender. Scenes are described explicitly
via a JSON file that specifies the shape/color/material of each object, mirroring
the sampling strategy defined in the paper.

Usage:
blender --background --python render_binding.py -- --scene_specs_path specs.json [args]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from datetime import datetime as dt
from typing import Any, Dict, Iterable, List, Sequence

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False

if INSIDE_BLENDER:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)
    try:
        import utils
    except ImportError:
        print("\nERROR")
        print("Unable to import utils.py inside Blender. Ensure that the "
              "binding/3d_dataset/blender directory is on Blender's Python path.")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_specs_path",
        required=True,
        help="Path to a JSON file containing serialized scene specifications.",
    )
    parser.add_argument(
        "--base_scene_blendfile",
        default="data/base_scene.blend",
        help="Base blender file on which all scenes are based.",
    )
    parser.add_argument(
        "--properties_json",
        default="data/properties.json",
        help="JSON file defining shapes, colors, materials, and sizes.",
    )
    parser.add_argument("--shape_dir", default="data/shapes", help="Directory containing shape .blend files.")
    parser.add_argument("--material_dir", default="data/materials", help="Directory containing material .blend files.")
    parser.add_argument(
        "--min_dist",
        default=0.25,
        type=float,
        help="Minimum allowed distance between object centers.",
    )
    parser.add_argument(
        "--margin",
        default=0.4,
        type=float,
        help="Minimum margin along cardinal directions between all objects.",
    )
    parser.add_argument(
        "--min_pixels_per_object",
        default=200,
        type=int,
        help="All objects will occupy at least this many pixels to avoid occlusion.",
    )
    parser.add_argument("--max_retries", default=50, type=int, help="Max attempts before restarting placement.")
    parser.add_argument("--split", default="3D", help="Dataset split stored in the emitted scene JSON.")
    parser.add_argument(
        "--width",
        default=512,
        type=int,
        help="Width (pixels) for rendered images.",
    )
    parser.add_argument(
        "--height",
        default=512,
        type=int,
        help="Height (pixels) for rendered images.",
    )
    parser.add_argument(
        "--render_num_samples",
        default=512,
        type=int,
        help="Number of samples for ray tracing.",
    )
    parser.add_argument("--render_min_bounces", default=8, type=int)
    parser.add_argument("--render_max_bounces", default=8, type=int)
    parser.add_argument(
        "--render_tile_size",
        default=256,
        type=int,
        help="Tile size for rendering.",
    )
    parser.add_argument(
        "--camera_jitter",
        default=0.5,
        type=float,
        help="Amount of random jitter to add to the camera.",
    )
    parser.add_argument(
        "--key_light_jitter",
        default=1.0,
        type=float,
        help="Amount of random jitter to add to the key light.",
    )
    parser.add_argument("--fill_light_jitter", default=1.0, type=float)
    parser.add_argument("--back_light_jitter", default=1.0, type=float)
    parser.add_argument(
        "--output_scene_file",
        default="scenes.json",
        help="Combined JSON file produced after all scenes are rendered.",
    )
    parser.add_argument(
        "--compute_device_type",
        default="CPU",
        help="Cycles compute device to use (CPU, CUDA, OPTIX, HIP, METAL, ONEAPI).",
    )
    parser.add_argument(
        "--num_images",
        default=0,
        type=int,
        help="Optional limit on the number of scenes to render; 0 renders all specs.",
    )
    parser.add_argument(
        "--start_idx",
        default=0,
        type=int,
        help="Optional offset into the scene specs (for sharded rendering).",
    )
    parser.add_argument(
        "--version",
        default="1.0",
        help="Version string stored in the combined scene JSON.",
    )
    parser.add_argument("--license", default="CC-BY 4.0", help="License string for the combined scene JSON.")
    parser.add_argument(
        "--date",
        default=dt.today().strftime("%m/%d/%Y"),
        help="Date stored in the combined scene JSON.",
    )
    return parser.parse_args(utils.extract_args() if INSIDE_BLENDER else None)


def main() -> None:
    args = parse_args()
    with open(args.scene_specs_path, "r") as f:
        spec_bundle = json.load(f)
    scene_specs = spec_bundle.get("scenes", [])
    start = args.start_idx
    end = len(scene_specs) if args.num_images <= 0 else min(len(scene_specs), start + args.num_images)
    selected_specs = scene_specs[start:end]
    if not selected_specs:
        print("No scenes selected for rendering; exiting.")
        return

    all_scene_paths: List[str] = []
    os.makedirs(os.path.dirname(args.output_scene_file), exist_ok=True)
    for spec in selected_specs:
        scene_path = spec["output_scene"]
        all_scene_paths.append(scene_path)
        render_scene(args, scene_spec=spec)

    # Aggregate per-scene JSONs into one file for convenience.
    scenes = []
    for scene_path in all_scene_paths:
        with open(scene_path, "r") as f:
            scenes.append(json.load(f))
    payload = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": scenes,
    }
    with open(args.output_scene_file, "w") as f:
        json.dump(payload, f, indent=2)


def configure_cycles(args: argparse.Namespace) -> None:
    prefs = getattr(bpy.context, "preferences", None)
    if prefs is None:
        prefs = bpy.context.user_preferences
    cycles_prefs = prefs.addons["cycles"].preferences
    device_type = args.compute_device_type.upper()
    if device_type != "CPU":
        cycles_prefs.compute_device_type = device_type
        bpy.context.scene.cycles.device = "GPU"
    else:
        bpy.context.scene.cycles.device = "CPU"


def render_scene(args: argparse.Namespace, scene_spec: Dict[str, Any]) -> None:
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    configure_cycles(args)
    utils.load_materials(args.material_dir)

    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = scene_spec["output_image"]
    os.makedirs(os.path.dirname(render_args.filepath), exist_ok=True)
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    if hasattr(render_args, "tile_x"):
        render_args.tile_x = args.render_tile_size
        render_args.tile_y = args.render_tile_size
    else:
        cycles_settings = bpy.context.scene.cycles
        if hasattr(cycles_settings, "tile_x"):
            cycles_settings.tile_x = args.render_tile_size
            cycles_settings.tile_y = args.render_tile_size
        elif hasattr(cycles_settings, "tile_size"):
            cycles_settings.tile_size = args.render_tile_size
    world = bpy.data.worlds.get("World")
    cycles_world = getattr(world, "cycles", None) if world else None
    if cycles_world and hasattr(cycles_world, "sample_as_light"):
        # Blender <5.0 exposes this MIS toggle; newer versions removed it.
        cycles_world.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

    scene_struct = {
        "scene_id": scene_spec.get("scene_id"),
        "split": scene_spec.get("split", args.split),
        "image_filename": os.path.basename(scene_spec["output_image"]),
        "image_index": scene_spec.get("image_index", 0),
        "objects": [],
        "directions": {},
        "metadata": scene_spec.get("metadata", {}),
    }

    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L: float) -> float:
        return 2.0 * L * (random.random() - 0.5)

    camera = bpy.data.objects["Camera"]
    if args.camera_jitter > 0:
        for i in range(3):
            camera.location[i] += rand(args.camera_jitter)

    camera_quat = camera.matrix_world.to_quaternion()
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera_quat * Vector((0, 0, -1))
    cam_left = camera_quat * Vector((-1, 0, 0))
    cam_up = camera_quat * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()
    utils.delete_object(plane)

    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    for light_name, jitter in [
        ("Lamp_Key", args.key_light_jitter),
        ("Lamp_Back", args.back_light_jitter),
        ("Lamp_Fill", args.fill_light_jitter),
    ]:
        if jitter > 0:
            for i in range(3):
                bpy.data.objects[light_name].location[i] += rand(jitter)

    objects, blender_objects = add_specified_objects(scene_struct, scene_spec["objects"], args, camera)
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    os.makedirs(os.path.dirname(scene_spec["output_scene"]), exist_ok=True)
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as exc:
            print(exc)

    with open(scene_spec["output_scene"], "w") as f:
        json.dump(scene_struct, f, indent=2)

    blend_path = scene_spec.get("output_blendfile")
    if blend_path:
        os.makedirs(os.path.dirname(blend_path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)


def load_properties(properties_path: str) -> Dict[str, Dict[str, Any]]:
    with open(properties_path, "r") as f:
        return json.load(f)


def add_specified_objects(scene_struct, object_specs, args, camera):
    properties = load_properties(args.properties_json)
    color_lookup = {
        name: [float(c) / 255.0 for c in values] + [1.0]
        for name, values in properties["colors"].items()
    }
    sizes = properties["sizes"]
    materials = properties["materials"]
    shapes = properties["shapes"]

    positions = []
    objects = []
    blender_objects = []
    for spec in object_specs:
        size_key = spec.get("size", next(iter(sizes)))
        radius = sizes.get(size_key, list(sizes.values())[0])
        color_key = spec["color"]
        rgba = color_lookup[color_key]
        shape_key = spec["shape"]
        blend_shape = shapes.get(shape_key)
        if blend_shape is None:
            raise ValueError(f"Shape '{shape_key}' missing from properties.")
        material_key = spec.get("material", next(iter(materials)))
        material_name = materials.get(material_key)
        if material_name is None:
            raise ValueError(f"Material '{material_key}' missing from properties.")

        num_tries = 0
        while True:
            num_tries += 1
            if num_tries > args.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_specified_objects(scene_struct, object_specs, args, camera)
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            if not _placement_ok((x, y, radius), positions, scene_struct, args):
                continue
            break

        theta = spec.get("rotation", 360.0 * random.random())
        utils.add_object(args.shape_dir, blend_shape, radius, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, radius))
        utils.add_material(material_name, Color=rgba)
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append(
            {
                "shape": shape_key,
                "size": size_key,
                "material": material_key,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "color": color_key,
                "metadata": spec.get("metadata", {}),
            }
        )

    all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
    if not all_visible:
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_specified_objects(scene_struct, object_specs, args, camera)
    return objects, blender_objects


def _placement_ok(candidate, positions, scene_struct, args) -> bool:
    x, y, r = candidate
    for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
            return False
        for direction_name in ["left", "right", "front", "behind"]:
            direction_vec = scene_struct["directions"][direction_name]
            margin = dx * direction_vec[0] + dy * direction_vec[1]
            if 0 < margin < args.margin:
                return False
    return True


def compute_all_relationships(scene_struct, eps=0.2):
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name in ("above", "below"):
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    f, path = tempfile.mkstemp(suffix=".png")
    os.close(f)
    render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    pixels = list(img.pixels)
    color_count = Counter(
        (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]) for i in range(0, len(pixels), 4)
    )
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


def render_shadeless(blender_objects, path="flat.png"):
    render_args = bpy.context.scene.render
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    render_args.filepath = path
    render_args.engine = "BLENDER_RENDER"
    render_args.use_antialiasing = False

    utils.set_layer(bpy.data.objects["Lamp_Key"], 2)
    utils.set_layer(bpy.data.objects["Lamp_Fill"], 2)
    utils.set_layer(bpy.data.objects["Lamp_Back"], 2)
    utils.set_layer(bpy.data.objects["Ground"], 2)

    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials["Material"]
        mat.name = f"Material_{i}"
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors:
                break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    bpy.ops.render.render(write_still=True)

    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    utils.set_layer(bpy.data.objects["Lamp_Key"], 0)
    utils.set_layer(bpy.data.objects["Lamp_Fill"], 0)
    utils.set_layer(bpy.data.objects["Lamp_Back"], 0)
    utils.set_layer(bpy.data.objects["Ground"], 0)
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing


if __name__ == "__main__":
    if INSIDE_BLENDER:
        main()
    else:
        print("This script must be executed from within Blender, e.g.:")
        print("blender --background --python render_binding.py -- --scene_specs_path specs.json")
