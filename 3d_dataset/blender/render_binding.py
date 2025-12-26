"""
Render binding-problem stimuli using Blender 5.x.

Scenes are described explicitly via a JSON file that specifies the shape/color/material
of each object.

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
from typing import Any, Dict, List, Tuple

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
    import utils  # noqa: E402


def log(msg: str) -> None:
    """Emit progress messages that survive Blender's minimal stdout buffering."""
    print(f"[render_binding] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_specs_path", required=True)
    parser.add_argument("--base_scene_blendfile", default="data/base_scene.blend")
    parser.add_argument("--properties_json", default="data/properties.json")
    parser.add_argument("--shape_dir", default="data/shapes")
    parser.add_argument("--material_dir", default="data/materials")

    parser.add_argument("--min_dist", default=0.25, type=float)
    parser.add_argument("--margin", default=0.4, type=float)
    parser.add_argument("--min_pixels_per_object", default=10, type=int)
    parser.add_argument("--max_retries", default=10, type=int)

    parser.add_argument("--split", default="3D")
    parser.add_argument("--width", default=100, type=int)
    parser.add_argument("--height", default=100, type=int)

    parser.add_argument("--render_num_samples", default=100, type=int)
    parser.add_argument("--render_min_bounces", default=8, type=int)
    parser.add_argument("--render_max_bounces", default=8, type=int)
    parser.add_argument(
        "--render_tile_size",
        "--render-tile-size",
        dest="render_tile_size",
        default=0,
        type=int,
        help="Tile size in pixels (0 lets Blender decide).",
    )

    parser.add_argument("--camera_jitter", default=0.5, type=float)
    parser.add_argument("--key_light_jitter", default=1.0, type=float)
    parser.add_argument("--fill_light_jitter", default=1.0, type=float)
    parser.add_argument("--back_light_jitter", default=1.0, type=float)
    parser.add_argument(
        "--max_layout_attempts",
        default=20,
        type=int,
        help="Number of times to rebuild a layout before rendering anyway.",
    )

    parser.add_argument("--output_scene_file", default="scenes.json")
    parser.add_argument(
        "--compute_device_type",
        default="CPU",
        help="CPU, CUDA, OPTIX, HIP, METAL, ONEAPI (depending on your build/GPU).",
    )

    parser.add_argument("--num_images", default=0, type=int)
    parser.add_argument("--start_idx", default=0, type=int)

    parser.add_argument("--version", default="1.0")
    parser.add_argument("--license", default="CC-BY 4.0")
    parser.add_argument("--date", default=dt.today().strftime("%m/%d/%Y"))

    argv = utils.extract_args() if INSIDE_BLENDER else None
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    log(f"Loaded {args.scene_specs_path} (num_images={args.num_images or 'all'})")

    with open(args.scene_specs_path, "r") as f:
        spec_bundle = json.load(f)

    scene_specs = spec_bundle.get("scenes", [])
    start = args.start_idx
    end = len(scene_specs) if args.num_images <= 0 else min(len(scene_specs), start + args.num_images)
    selected = scene_specs[start:end]
    if not selected:
        print("No scenes selected; exiting.")
        return

    os.makedirs(os.path.dirname(args.output_scene_file) or ".", exist_ok=True)

    per_scene_paths: List[str] = []
    for idx, spec in enumerate(selected, start=start):
        log(f"Rendering scene {idx} -> {spec.get('output_image')}")
        per_scene_paths.append(spec["output_scene"])
        render_scene(args, scene_spec=spec)

    scenes = []
    for p in per_scene_paths:
        with open(p, "r") as f:
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
    log(f"Wrote combined scene file to {args.output_scene_file}")


def configure_cycles(args: argparse.Namespace) -> None:
    log(f"Configuring Cycles for device {args.compute_device_type}")
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Cycles prefs
    prefs = bpy.context.preferences
    if "cycles" not in prefs.addons:
        print("WARNING: Cycles addon not enabled; rendering may fail.")
        return

    cycles_prefs = prefs.addons["cycles"].preferences
    dev = (args.compute_device_type or "CPU").upper()

    if dev == "CPU":
        scene.cycles.device = "CPU"
        log("Using CPU rendering")
        log(f"Cycles device state: {scene.cycles.device}")
        return

    # Set compute device type and populate devices list if supported
    try:
        cycles_prefs.compute_device_type = dev
        if hasattr(cycles_prefs, "get_devices"):
            cycles_prefs.get_devices()
        scene.cycles.device = "GPU"
        log(f"Configured GPU device mode {dev}")
        log(f"Cycles device state: {scene.cycles.device}")
    except Exception as e:
        print(f"WARNING: Failed to configure GPU device '{dev}', falling back to CPU. Error: {e}")
        scene.cycles.device = "CPU"
        log("Falling back to CPU rendering")
        log(f"Cycles device state: {scene.cycles.device}")


def render_scene(args: argparse.Namespace, scene_spec: Dict[str, Any]) -> None:
    bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)
    log(f"Opened base blend; preparing scene {scene_spec.get('scene_id')}")

    configure_cycles(args)
    utils.load_materials(args.material_dir)

    scene = bpy.context.scene
    render_args = scene.render
    cycles = scene.cycles

    render_args.filepath = scene_spec["output_image"]
    os.makedirs(os.path.dirname(render_args.filepath) or ".", exist_ok=True)

    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100

    tile = int(getattr(args, "render_tile_size", 0) or 0)

    # Older Cycles versions
    if hasattr(render_args, "tile_x") and tile > 0:
        render_args.tile_x = tile
        render_args.tile_y = tile

    # Newer Cycles versions (some builds)
    elif hasattr(bpy.context.scene.cycles, "tile_size") and tile > 0:
        bpy.context.scene.cycles.tile_size = tile
    cycles.samples = args.render_num_samples

    # Bounce property names differ across versions; handle both
    if hasattr(cycles, "transparent_min_bounces"):
        cycles.transparent_min_bounces = args.render_min_bounces
    elif hasattr(cycles, "min_transparent_bounces"):
        cycles.min_transparent_bounces = args.render_min_bounces

    if hasattr(cycles, "transparent_max_bounces"):
        cycles.transparent_max_bounces = args.render_max_bounces
    elif hasattr(cycles, "max_transparent_bounces"):
        cycles.max_transparent_bounces = args.render_max_bounces

    # (Optional) some versions have blur_glossy
    if hasattr(cycles, "blur_glossy"):
        cycles.blur_glossy = 2.0

    # Build scene struct
    scene_struct = {
        "scene_id": scene_spec.get("scene_id"),
        "split": scene_spec.get("split", args.split),
        "image_filename": os.path.basename(scene_spec["output_image"]),
        "image_index": scene_spec.get("image_index", 0),
        "objects": [],
        "directions": {},
        "metadata": scene_spec.get("metadata", {}),
    }

    # Temporary plane to compute directions
    bpy.ops.mesh.primitive_plane_add(size=10)
    plane = bpy.context.object

    def rand(L: float) -> float:
        return 2.0 * L * (random.random() - 0.5)

    camera = bpy.data.objects.get("Camera")
    if camera is None:
        raise RuntimeError("Camera object named 'Camera' not found in base scene.")

    if args.camera_jitter > 0:
        for i in range(3):
            camera.location[i] += rand(args.camera_jitter)

    camera_quat = camera.matrix_world.to_quaternion()
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera_quat @ Vector((0, 0, -1))
    cam_left = camera_quat @ Vector((-1, 0, 0))
    cam_up = camera_quat @ Vector((0, 1, 0))

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

    # Jitter lights (only if present)
    for light_name, jitter in [
        ("Lamp_Key", args.key_light_jitter),
        ("Lamp_Back", args.back_light_jitter),
        ("Lamp_Fill", args.fill_light_jitter),
    ]:
        light_obj = bpy.data.objects.get(light_name)
        if light_obj and jitter > 0:
            for i in range(3):
                light_obj.location[i] += rand(jitter)

    objects, blender_objects = add_specified_objects(scene_struct, scene_spec["objects"], args, camera)
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    os.makedirs(os.path.dirname(scene_spec["output_scene"]) or ".", exist_ok=True)

    log(f"Rendering image to {scene_spec['output_image']}")
    bpy.ops.render.render(write_still=True)
    log(f"Finished render {scene_spec.get('scene_id')}")

    with open(scene_spec["output_scene"], "w") as f:
        json.dump(scene_struct, f, indent=2)

    blend_path = scene_spec.get("output_blendfile")
    if blend_path:
        os.makedirs(os.path.dirname(blend_path) or ".", exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)


def load_properties(properties_path: str) -> Dict[str, Any]:
    with open(properties_path, "r") as f:
        return json.load(f)


def _placement_ok(candidate: Tuple[float, float, float], positions, scene_struct, args) -> bool:
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


def add_specified_objects(scene_struct, object_specs, args, camera):
    log(f"Placing {len(object_specs)} objects")
    properties = load_properties(args.properties_json)

    color_lookup = {
        name: [float(c) / 255.0 for c in rgb] + [1.0]
        for name, rgb in properties["colors"].items()
    }
    sizes = properties["sizes"]
    materials = properties["materials"]
    shapes = properties["shapes"]

    def cleanup(created_objs):
        for obj in created_objs:
            utils.delete_object(obj)

    max_layout_attempts = max(1, int(getattr(args, "max_layout_attempts", 20)))
    objects = []
    blender_objects = []
    created_objects = []

    for attempt in range(1, max_layout_attempts + 1):
        if max_layout_attempts > 1:
            log(f"Layout attempt {attempt}/{max_layout_attempts}")
        positions = []
        objects = []
        blender_objects = []
        created_objects = []
        success = True

        for idx, spec in enumerate(object_specs):
            size_key = spec.get("size", next(iter(sizes)))
            radius = sizes.get(size_key, list(sizes.values())[0])

            color_key = spec["color"]
            rgba = color_lookup[color_key]

            shape_key = spec["shape"]
            blend_shape = shapes.get(shape_key)
            if blend_shape is None:
                raise ValueError(f"Shape '{shape_key}' missing from properties.json")

            material_key = spec.get("material", next(iter(materials)))
            material_name = materials.get(material_key)
            if material_name is None:
                raise ValueError(f"Material '{material_key}' missing from properties.json")

            num_tries = 0
            placed = False
            while num_tries < args.max_retries:
                num_tries += 1
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                if _placement_ok((x, y, radius), positions, scene_struct, args):
                    placed = True
                    break

            if not placed:
                log(f"  [{idx}] placement retries exhausted (attempt {attempt}); restarting layout")
                success = False
                break

            theta = spec.get("rotation", 360.0 * random.random())
            obj, new_objs = utils.add_object(args.shape_dir, blend_shape, radius, (x, y), theta=theta)
            log(
                f"  [{idx}] placed {shape_key} ({color_key}/{material_key}/{size_key})"
                f" at ({x:.2f}, {y:.2f}) r={radius:.2f}"
            )
            created_objects.extend(new_objs)

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
                    "rotation": float(theta),
                    "pixel_coords": pixel_coords,
                    "color": color_key,
                    "metadata": spec.get("metadata", {}),
                }
            )

        if not success:
            if attempt < max_layout_attempts:
                cleanup(created_objects)
                continue
            log("Layout attempts exhausted; proceeding with partially placed objects.")
            return objects, blender_objects

        if check_visibility(blender_objects, args.min_pixels_per_object):
            log("Visibility check passed")
            return objects, blender_objects

        log("Visibility check failed; retrying layout")
        if attempt < max_layout_attempts:
            cleanup(created_objects)
            continue
        log("Layout attempts exhausted; proceeding despite visibility failure.")
        return objects, blender_objects

    # Fallback (should not hit because of returns above)
    return objects, blender_objects


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
                diff = [coords2[k] - coords1[k] for k in (0, 1, 2)]
                dot = sum(diff[k] * direction_vec[k] for k in (0, 1, 2))
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def _make_emission_material(name: str, rgba):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = rgba
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def render_shadeless(blender_objects, path="flat.png"):
    """
    Render a flat mask image to test visibility. Works in Blender 5.x.
    Uses Workbench engine + per-object emission materials + hide_render on lights/ground.
    """
    scene = bpy.context.scene
    render_args = scene.render

    old_engine = render_args.engine
    old_filepath = render_args.filepath

    render_args.engine = "BLENDER_WORKBENCH"
    render_args.filepath = path

    # Disable AA if available
    if hasattr(scene, "display") and hasattr(scene.display, "render_aa"):
        try:
            scene.display.render_aa = "OFF"
        except Exception:
            pass

    # Hide lights/ground from render
    to_hide = []
    for name in ["Lamp_Key", "Lamp_Fill", "Lamp_Back", "Ground"]:
        obj = bpy.data.objects.get(name)
        if obj:
            to_hide.append((obj, obj.hide_render))
            obj.hide_render = True

    # Assign unique emission materials to each object
    object_colors = set()
    old_mats = []

    for i, obj in enumerate(blender_objects):
        old_mats.append([m for m in obj.data.materials])

        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors:
                break
        object_colors.add((r, g, b))
        rgba = (r, g, b, 1.0)

        obj.data.materials.clear()
        obj.data.materials.append(_make_emission_material(f"MaskMat_{i}", rgba))

    bpy.ops.render.render(write_still=True)

    # Restore materials
    for obj, mats in zip(blender_objects, old_mats):
        obj.data.materials.clear()
        for m in mats:
            obj.data.materials.append(m)

    # Restore hide_render
    for obj, old in to_hide:
        obj.hide_render = old

    render_args.engine = old_engine
    render_args.filepath = old_filepath

    return object_colors


def check_visibility(blender_objects, min_pixels_per_object: int) -> bool:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    render_shadeless(blender_objects, path=path)

    img = bpy.data.images.load(path)
    pixels = list(img.pixels)

    # cleanup image datablock to avoid memory growth
    bpy.data.images.remove(img)

    os.remove(path)

    color_count = Counter(
        (pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3])
        for i in range(0, len(pixels), 4)
    )

    # Expect background + one color per object
    if len(color_count) != len(blender_objects) + 1:
        return False

    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


if __name__ == "__main__":
    if INSIDE_BLENDER:
        main()
    else:
        print("Run from Blender, e.g.:")
        print("blender --background --python render_binding.py -- --scene_specs_path specs.json")
