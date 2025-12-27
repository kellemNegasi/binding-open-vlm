from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from itertools import product
import numpy as np
import colorsys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
log_folder = 'logs'
if not os.path.exists(log_folder):
  os.makedirs(log_folder)
timestamp = dt.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_folder, 'render_images_{}.log'.format(timestamp))
file_handler = logging.FileHandler(log_file)
logger.addHandler(file_handler)

INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError:
  INSIDE_BLENDER = False

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='blender_utils/base_scene.blend',
    help="Base blender file on which all scenes are based; includes ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='blender_utils/properties.json',
    help="JSON file defining objects, materials, sizes, and colors.")
parser.add_argument('--shape_dir', default='blender_utils/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='blender_utils/materials',
    help="Directory where .blend files for materials are stored")

# Settings for objects
parser.add_argument('--num_objects', default=20, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.2, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.1, type=float,
    help="Along all cardinal directions (left, right, front, back), all objects will be at least this distance apart.")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="All objects will have at least this many visible pixels in the final rendered images.")
parser.add_argument('--max_retries', default=500, type=int,
    help="The number of times to try placing an object before giving up and re-placing all objects.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="Prefix prepended to rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Split name stored in JSON and used in filenames.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="Directory where output images will be stored.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="Directory where output JSON scene structures will be stored.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--version', default='1.0',
    help="String stored in the \"version\" field of the output JSON")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String stored in the \"license\" field of the output JSON")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String stored in the \"date\" field of the output JSON")

# Rendering options
parser.add_argument('--width', default=640, type=int,
    help="Rendered image width in pixels")
parser.add_argument('--height', default=480, type=int,
    help="Rendered image height in pixels")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="Random jitter magnitude for key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="Random jitter magnitude for fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="Random jitter magnitude for back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="Random jitter magnitude for camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="Cycles samples for rendering")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="Minimum bounces for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="Maximum bounces for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="(Legacy) tile size; not used in Blender 5.0 Cycles in the same way.")
parser.add_argument('--task', default="counting", type=str,
    help="Task type controlling object/color choices.")
parser.add_argument('--num_features', default=2, type=int,
    help="Number of unique features for binding tasks")
parser.add_argument('--override_output_dir', default=False, action='store_true',
    help="Whether to override the output directory")


def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)

  if args.override_output_dir:
    output_image_dir = args.output_image_dir.replace("images", args.task + "/images")
    output_scene_dir = args.output_scene_dir.replace("scenes", args.task + "/scenes")
  else:
    output_image_dir = args.output_image_dir
    output_scene_dir = args.output_scene_dir

  img_template = os.path.join(output_image_dir, img_template)
  scene_template = os.path.join(output_scene_dir, scene_template)

  if not os.path.isdir(output_image_dir):
    os.makedirs(output_image_dir)
  if not os.path.isdir(output_scene_dir):
    os.makedirs(output_scene_dir)

  all_scene_paths = []
  num_images = args.num_images
  num_objects = args.num_objects

  objects_features = None
  if args.task == 'binding':
    with open(args.properties_json, 'r') as f:
      properties = json.load(f)
      color_name_to_rgba = {}
      for name, rgb in properties['colors'].items():
        rgba = [np.round(float(c) / 255.0, 2) for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
      object_mapping = [(v, k) for k, v in properties['shapes'].items()]

    task_conditions = list(product(range(1, num_objects + 1), range(1, num_objects + 1)))
    condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
    counts, count_freq = np.unique(condition_feature_counts, return_counts=True)

    num_features_dict = {}
    for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):
      num_features_dict[n_features] = [(n_shapes, n_colors)] + num_features_dict.get(n_features, [])

    objects_features = []
    for n_shapes, n_colors in num_features_dict[args.num_features]:
      n_trials = int(np.ceil(args.num_images / count_freq[counts == args.num_features][0]))
      shape_names = np.array([shape for shape, _ in object_mapping])
      color_names = np.array([color for color, _ in color_name_to_rgba.items()])
      for _ in range(n_trials):
        n_shapes = min(n_shapes, len(shape_names))
        n_shapes = min(n_shapes, n_objects)
        unique_shape_inds = np.random.choice(len(shape_names), n_shapes, replace=False)
        shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, num_objects - n_shapes, replace=True)])
        n_colors = min(n_colors, len(color_names))
        n_colors = min(n_colors, n_objects)
        unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)
        color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, num_objects - n_colors, replace=True)])
        colors = color_names[color_inds]
        shapes = shape_names[shape_inds]
        trial_features = list(zip(shapes, colors))
        objects_features.append(trial_features)
    random.shuffle(objects_features)

  for i in range(num_images):
    logger.info('\nRendering image %d / %d' % (i, num_images))
    num_objects = args.num_objects
    index = i + 100 * num_objects + args.start_idx

    if args.task == 'binding':
      index = i + 100 * args.num_features + args.start_idx
      num_objects = args.num_objects

    img_path = img_template % (index)
    scene_path = scene_template % (index)
    all_scene_paths.append(scene_path)

    logger.info('\nImage path: %s, index: %d' % (img_path, index))
    logger.info('\nRendering scene %d with %d objects' % (index, num_objects))

    render_scene(
      args,
      num_objects=num_objects,
      output_index=index,
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      objects_features=objects_features[i] if (args.task == 'binding' and objects_features is not None) else None
    )

  # Combine per-scene JSON files into one
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))

  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)


def generate_isoluminant_colors(num_colors, saturation=1, lightness=0.8, mu=0.5, sigma=0.1, mode="min"):
  if mode == "max":
    hues = np.linspace(0, 1, num_colors, endpoint=False)
  elif mode == "intermediate":
    mu = np.random.uniform(0, 1)
    sigma = 1e-1
    hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
  elif mode == "min":
    mu = np.random.uniform(0, 1)
    sigma = 1e-4
    hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
  else:
    raise ValueError('Unrecognized mode')
  hsl_colors = [(hue, saturation, lightness) for hue in hues]
  rgb_colors = [colorsys.hsv_to_rgb(*color) for color in hsl_colors]
  return rgb_colors


def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    objects_features=None
  ):

  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  load_materials(args.material_dir)

  scene = bpy.context.scene
  render_args = scene.render

  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100

  # NOTE: tile_x / tile_y are not reliably available/used in modern Cycles, so we omit them.

  # Cycles settings
  try:
    bpy.data.worlds['World'].cycles.sample_as_light = True
  except Exception:
    pass

  scene.cycles.blur_glossy = 2.0
  scene.cycles.samples = args.render_num_samples
  # transparent bounces
  if hasattr(scene.cycles, "transparent_min_bounces"):
      scene.cycles.transparent_min_bounces = args.render_min_bounces  # old Blender
  # else: Blender 5.x doesn't have it; ignore

  if hasattr(scene.cycles, "transparent_max_bounces"):
    scene.cycles.transparent_max_bounces = args.render_max_bounces

  # Turn off denoise for deterministic pixel counting renders (if present)
  if hasattr(scene.cycles, "use_denoising"):
    scene.cycles.use_denoising = False

  scene_struct = {
    'split': output_split,
    'image_index': output_index,
    'image_filename': os.path.basename(output_image),
    'objects': [],
    'directions': {},
  }

  # Add temp plane to compute directions
  bpy.ops.mesh.primitive_plane_add(size=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Camera jitter
  if args.camera_jitter > 0:
    cam = bpy.data.objects.get('Camera')
    if cam is not None:
      for i in range(3):
        cam.location[i] += rand(args.camera_jitter)

  camera = bpy.data.objects['Camera']

  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() @ Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() @ Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  delete_object(plane)

  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Light jitter (names must match your base scene)
  if args.key_light_jitter > 0 and bpy.data.objects.get('Lamp_Key') is not None:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0 and bpy.data.objects.get('Lamp_Back') is not None:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0 and bpy.data.objects.get('Lamp_Fill') is not None:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, objects_features)
  logger.info('\nObjects added')

  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)

  logger.info('\nStarting rendering')
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      logger.info('\nRendering error %s' % e)
      print(e)

  logger.info('\nRendering done')
  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)


def add_random_objects(scene_struct, num_objects, args, camera, objects_features=None):
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [np.round(float(c) / 255.0, 2) for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  if args.task == "counting_min_distinctiveness":
    color_name_to_rgba = {}
    colors = generate_isoluminant_colors(num_objects, mode="min")
    for i in range(num_objects):
      color_name_to_rgba["color_%s" % str(i)] = list(colors[0]) + [1.0]
  elif args.task == "counting_intermediate":
    color_name_to_rgba = {}
    colors = generate_isoluminant_colors(num_objects, mode="intermediate")
    for i in range(num_objects):
      color_name_to_rgba["color_%s" % str(i)] = list(colors[i]) + [1.0]
  elif args.task == "counting_max_distinctiveness":
    color_name_to_rgba = {}
    colors = generate_isoluminant_colors(num_objects, mode="max")
    for i in range(num_objects):
      color_name_to_rgba["color_%s" % str(i)] = list(colors[i]) + [1.0]

  positions = []
  objects = []
  blender_objects = []

  for i in range(num_objects):
    max_retries = args.max_retries + 10 * i
    logger.info('\nAdding object %d of %d\n' % (i, num_objects))

    size_name, r = random.choice(size_mapping)

    num_tries = 0
    while True:
      num_tries += 1
      if num_tries > max_retries:
        logger.info("Max retries exceeded: %d" % max_retries)
        for obj in blender_objects:
          delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera, objects_features)

      radius = random.uniform(0, 4.5)
      alpha = random.uniform(0, 2 * math.pi)
      x, y = radius * math.cos(alpha), radius * math.sin(alpha)
      logger.info('\nTrying to place object at %f, %f, within %f' % (x, y, radius))

      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break
        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            margins_good = False
            break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

    # Choose random color and shape
    if args.task == 'binding':
      obj_name, obj_name_out = objects_features[i][0], dict(object_mapping)[objects_features[i][0]]
      color_name, rgba = objects_features[i][1], dict(color_name_to_rgba)[objects_features[i][1]]
    elif args.task == 'counting':
      obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
      color_name, rgba = "black", dict(color_name_to_rgba)["black"]
    elif args.task in ('counting_min_distinctiveness', 'counting_intermediate', 'counting_max_distinctiveness'):
      obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
      color_name, rgba = "color_" + str(i), dict(color_name_to_rgba)["color_" + str(i)]
    elif args.task == 'popout':
      if i == 0:
        obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
        color_name, rgba = "red", dict(color_name_to_rgba)["red"]
      else:
        obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
        color_name, rgba = "green", dict(color_name_to_rgba)["green"]
    elif args.task == 'popout_counterfactual':
      obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
      color_name, rgba = "green", dict(color_name_to_rgba)["green"]
    elif args.task == 'search':
      if i == 0:
        obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
        color_name, rgba = "red", dict(color_name_to_rgba)["red"]
      elif i % 2 == 0:
        obj_name, obj_name_out = "SmoothCube_v2", dict(object_mapping)["SmoothCube_v2"]
        color_name, rgba = "red", dict(color_name_to_rgba)["red"]
      else:
        obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
        color_name, rgba = "green", dict(color_name_to_rgba)["green"]
    elif args.task == 'search_counterfactual':
      if i % 2 == 0:
        obj_name, obj_name_out = "SmoothCube_v2", dict(object_mapping)["SmoothCube_v2"]
        color_name, rgba = "red", dict(color_name_to_rgba)["red"]
      else:
        obj_name, obj_name_out = "Sphere", dict(object_mapping)["Sphere"]
        color_name, rgba = "green", dict(color_name_to_rgba)["green"]
    elif args.task == 'print_all_shapes':
      obj_name, obj_name_out = object_mapping[i]
      color_name, rgba = "green", dict(color_name_to_rgba)["green"]
    else:
      raise ValueError('Unrecognized task')

    logger.info('\nAdding object: %s, %s, %s' % (obj_name_out, color_name, rgba))

    if obj_name == 'SmoothCube_v2':
      r *= 0.9

    theta_deg = 360.0 * random.random()

    # Add object (Blender 5-safe append/link)
    add_object(args.shape_dir, obj_name, r, (x, y), theta=theta_deg)
    obj = bpy.context.view_layer.objects.active
    blender_objects.append(obj)
    positions.append((x, y, r))

    mat_name, mat_name_out = random.choice(material_mapping)
    add_material(mat_name, Color=rgba, logger=logger)

    pixel_coords = get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta_deg,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    logger.info('\nSome objects are occluded; replacing objects')
    for obj in blender_objects:
      delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera, objects_features)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name in ('above', 'below'):
      continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2:
          continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  f, path = tempfile.mkstemp(suffix='.png')
  os.close(f)

  object_colors = render_shadeless(blender_objects, path=path)

  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  logger.info('\nColor count: %s' % color_count)

  # cleanup
  try:
    bpy.data.images.remove(img)
  except Exception:
    pass
  try:
    os.remove(path)
  except Exception:
    pass

  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Blender 5.0 replacement for 2.7x "shadeless + BLENDER_RENDER":
  - Render with Cycles
  - Hide lights/ground from render
  - Assign unique EMISSION materials (flat colors)
  """
  scene = bpy.context.scene
  render_args = scene.render

  old_filepath = render_args.filepath
  old_engine = render_args.engine

  # Cache some Cycles settings we’ll restore
  old_samples = getattr(scene.cycles, "samples", None)
  old_denoise = getattr(scene.cycles, "use_denoising", None)

  render_args.filepath = path
  render_args.engine = 'CYCLES'

  # Make the pass fast and reduce blending/AA artifacts
  scene.cycles.samples = 1
  if hasattr(scene.cycles, "use_denoising"):
    scene.cycles.use_denoising = False

  # Hide lights + ground from render
  hidden = {}
  for name in ['Lamp_Key', 'Lamp_Fill', 'Lamp_Back', 'Ground']:
    obj = bpy.data.objects.get(name)
    if obj is not None:
      hidden[name] = obj.hide_render
      obj.hide_render = True

  object_colors = set()
  old_materials = []

  for i, obj in enumerate(blender_objects):
    old_mat = obj.data.materials[0] if obj.data.materials else None
    old_materials.append(old_mat)

    mat = bpy.data.materials.new(name=f"FlatMat_{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    emit = nodes.new(type="ShaderNodeEmission")

    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors:
        break
    object_colors.add((r, g, b))

    emit.inputs["Color"].default_value = (r, g, b, 1.0)
    emit.inputs["Strength"].default_value = 1.0
    links.new(emit.outputs["Emission"], out.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)

  bpy.ops.render.render(write_still=True)

  # Restore object materials
  for old_mat, obj in zip(old_materials, blender_objects):
    obj.data.materials.clear()
    if old_mat is not None:
      obj.data.materials.append(old_mat)

  # Restore hide_render
  for name, old_state in hidden.items():
    obj = bpy.data.objects.get(name)
    if obj is not None:
      obj.hide_render = old_state

  # Restore render settings
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  if old_samples is not None:
    scene.cycles.samples = old_samples
  if old_denoise is not None and hasattr(scene.cycles, "use_denoising"):
    scene.cycles.use_denoising = old_denoise

  return object_colors


def extract_args(input_argv=None):
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


def delete_object(obj):
  """Delete a specified blender object (Blender 2.8+ / 5.0 compatible)."""
  view_layer = bpy.context.view_layer
  bpy.ops.object.select_all(action='DESELECT')
  obj.select_set(True)
  view_layer.objects.active = obj
  bpy.ops.object.delete(use_global=False)


def get_camera_coords(cam, pos):
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  Blender 5 compatible object import/link:
  - Loads object datablock from .blend
  - Links it into the current collection
  - Applies transforms without context-sensitive operators
  """
  filepath = os.path.join(object_dir, f'{name}.blend')

  with bpy.data.libraries.load(filepath, link=False) as (data_from, data_to):
    src_name = name if name in data_from.objects else (data_from.objects[0] if data_from.objects else None)
    if src_name is None:
      raise RuntimeError(f"No objects found in {filepath}")
    data_to.objects = [src_name]

  obj = data_to.objects[0]
  if obj is None:
    raise RuntimeError(f"Failed to load object '{name}' from {filepath}")

  # Ensure unique name
  base = obj.name
  count = sum(o.name.startswith(base) for o in bpy.data.objects)
  obj.name = f"{base}_{count}"

  # Link into the current collection
  bpy.context.collection.objects.link(obj)

  # Apply transforms
  x, y = loc
  obj.scale = (scale, scale, scale)
  obj.location = (x, y, scale)

  # theta given in degrees in original code; convert to radians for Blender
  obj.rotation_euler[2] = math.radians(theta)

  # Make active
  bpy.context.view_layer.objects.active = obj


def load_materials(material_dir):
  """
  Load materials from a directory. Each .blend contains a NodeTree named X with a "Color" input.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'):
      continue
    name = os.path.splitext(fn)[0]

    # Append the node group by filepath + directory + filename (more robust than legacy single-string append)
    blend_path = os.path.join(material_dir, fn)
    directory = os.path.join(blend_path, 'NodeTree')
    filepath = os.path.join(directory, name)
    bpy.ops.wm.append(filepath=filepath, directory=directory, filename=name)


def add_material(name, **properties):
  """
  Create a new material and assign it to the active object.
  "name" must be a node group already loaded into bpy.data.node_groups.
  """
  obj = bpy.context.active_object
  if obj is None:
    obj = bpy.context.view_layer.objects.active
  if obj is None:
    raise RuntimeError("No active object found to assign a material.")

  # Clear existing materials
  if hasattr(obj.data, "materials"):
    obj.data.materials.clear()

  mat_count = len(bpy.data.materials)
  mat = bpy.data.materials.new(name=f"Material_{mat_count}")
  mat.use_nodes = True

  obj.data.materials.append(mat)

  # Ensure an output node exists
  output_node = None
  for n in mat.node_tree.nodes:
    if n.type == 'OUTPUT_MATERIAL':
      output_node = n
      break
  if output_node is None:
    output_node = mat.node_tree.nodes.new('ShaderNodeOutputMaterial')

  # Add group node
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  if name not in bpy.data.node_groups:
    raise RuntimeError(f"Node group '{name}' not found. Did load_materials() load it?")
  group_node.node_tree = bpy.data.node_groups[name]

  _logger = properties.get('logger', None)

  for inp in group_node.inputs:
    if inp.name in properties:
      if _logger is not None:
        _logger.info("Setting input %s: %s" % (inp.name, properties[inp.name]))
      inp.default_value = properties[inp.name]

  # Connect Shader -> Surface
  mat.node_tree.links.new(
    group_node.outputs.get('Shader'),
    output_node.inputs.get('Surface')
  )


if __name__ == '__main__':
  if INSIDE_BLENDER:
    argv = extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python gen-blender.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python gen-blender.py --help')
