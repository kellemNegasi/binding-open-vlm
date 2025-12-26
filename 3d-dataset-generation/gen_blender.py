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
except ImportError as e:
  INSIDE_BLENDER = False

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='blender_utils/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='blender_utils/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
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
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=100, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=500, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")


# Rendering options
parser.add_argument('--width', default=640, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=480, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")
parser.add_argument('--task', default="counting", type=str,
    help="The task to render images for. This will determine the types of " +
         "objects, colors, materials, sizes, etc. that will be used.")
parser.add_argument('--num_features', default=2, type=int,
    help="The number of unique features to use for binding tasks")
parser.add_argument('--override_output_dir', default=False, action='store_true',
    help="Whether to override the output directory")

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
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
  if args.task == 'binding':
    with open(args.properties_json, 'r') as f:
      properties = json.load(f)
      color_name_to_rgba = {}
      for name, rgb in properties['colors'].items():
        rgba = [np.round(float(c) / 255.0, 2) for c in rgb] + [1.0]
        color_name_to_rgba[name] = rgba
      object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    task_conditions = list(product(range(1,num_objects+1), range(1,num_objects+1)))
    condition_feature_counts = np.vstack(task_conditions).sum(axis=1)
    counts, count_freq = np.unique(condition_feature_counts, return_counts=True)
    num_features_dict = {}
    for n_features, (n_shapes, n_colors) in zip(condition_feature_counts, task_conditions):
      num_features_dict[n_features] = [(n_shapes, n_colors)] + num_features_dict.get(n_features, [])
    objects_features = []
    for n_shapes, n_colors in num_features_dict[args.num_features]:
      n_trials = int(np.ceil(args.num_images / count_freq[counts==args.num_features][0]))
      shape_names = np.array([shape for shape, _ in object_mapping])
      color_names = np.array([color for color, _ in color_name_to_rgba.items()])
      for i in range(n_trials):
        unique_shape_inds = np.random.choice(len(shape_names), n_shapes, replace=False) # sample the unique shapes for the current trial.
        shape_inds = np.concatenate([unique_shape_inds, np.random.choice(unique_shape_inds, num_objects-n_shapes, replace=True)])
        unique_color_inds = np.random.choice(len(color_names), n_colors, replace=False)  # sample the unique colors for the current trial.
        color_inds = np.concatenate([unique_color_inds, np.random.choice(unique_color_inds, num_objects-n_colors, replace=True)])
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
      logger.info('\nImage path: %s, index: %d' % (img_path, index))
      scene_path = scene_template % (index)
      all_scene_paths.append(scene_path)
      logger.info('\nRendering scene %d with %d objects' % (index, num_objects))
      render_scene(args,
        num_objects=num_objects,
        output_index=(index),
        output_split=args.split,
        output_image=img_path,
        output_scene=scene_path,
        objects_features=objects_features[i] if args.task == 'binding' else None
      )

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
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
  if mode=="max":
    hues = np.linspace(0, 1, num_colors, endpoint=False)
  elif mode == "intermediate":
    mu = np.random.uniform(0, 1)
    sigma = 1e-1
    hues = np.random.normal(loc=mu, scale=sigma, size=num_colors) % 1.0
  elif mode == "min":
    # sample mu
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

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
  render_args = bpy.context.scene.render
  render_args.engine = "CYCLES"
  render_args.filepath = output_image
  render_args.resolution_x = args.width
  render_args.resolution_y = args.height
  render_args.resolution_percentage = 100
  render_args.tile_x = args.render_tile_size
  render_args.tile_y = args.render_tile_size

  # Some CYCLES-specific stuff
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

  # Figure out the left, up, and behind directions along the plane and record
  # them in the scene structure
  camera = bpy.data.objects['Camera']
  plane_normal = plane.data.vertices[0].normal
  cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
  cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
  cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
  plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
  plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
  plane_up = cam_up.project(plane_normal).normalized()

  # Delete the plane; we only used it for normals anyway. The base scene file
  # contains the actual ground plane.
  delete_object(plane)

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera, objects_features)
  logger.info('\nObjects added')
  # Render the scene and dump the scene data structure
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
  """
  Add random objects to the current blender scene
  """

  # Load the property file
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
    max_retries = args.max_retries + 10*i
    logger.info('\nAdding object %d of %d\n' % (i, num_objects))
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > max_retries:
        logger.info("Max retries exceeded: %d" % max_retries)
        for obj in blender_objects:
          delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
      # while True:
      #   radius = 4
      #   x = random.uniform(-radius, radius)
      #   y = random.uniform(-radius, radius)
      #   if abs(x) + abs(y) < radius * 1.5:
      #     break
      radius = random.uniform(0, 4.5)
      alpha = random.uniform(0, 2 * math.pi)
      x, y = radius * math.cos(alpha), radius * math.sin(alpha)
      logger.info('\nTrying to place object at %f, %f, within %f' % (x, y, radius))
      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
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
            # logger.info('\nBroken margin: %s, %s, %s' % (margin, args.margin, direction_name))
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
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
    elif args.task == 'counting_min_distinctiveness' or args.task == 'counting_intermediate' or args.task == 'counting_max_distinctiveness':
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

    # For SmoothCube_v2, adjust the size a bit
    if obj_name == 'SmoothCube_v2':
      r *= 0.9

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    add_material(mat_name, Color=rgba, logger=logger)

    # Record data about the object in the scene data structure
    pixel_coords = get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    logger.info('\nSome objects are occluded; replacing objects')
    for obj in blender_objects:
      delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  logger.info('\nColor count: %s' % color_count)
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True


def render_shadeless(blender_objects, path='flat.png'):
  """
  Render a version of the scene with shading disabled and unique materials
  assigned to all objects, and return a set of all colors that should be in the
  rendered image. The image itself is written to path. This is used to ensure
  that all objects will be visible in the final rendered scene.
  """
  render_args = bpy.context.scene.render

  # Cache the render args we are about to clobber
  old_filepath = render_args.filepath
  old_engine = render_args.engine
  old_use_antialiasing = render_args.use_antialiasing

  # Override some render settings to have flat shading
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  set_layer(bpy.data.objects['Lamp_Key'], 2)
  set_layer(bpy.data.objects['Lamp_Fill'], 2)
  set_layer(bpy.data.objects['Lamp_Back'], 2)
  set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
    obj.data.materials[0] = mat

  # Move the lights and ground back to layer 0
  set_layer(bpy.data.objects['Lamp_Key'], 0)
  set_layer(bpy.data.objects['Lamp_Fill'], 0)
  set_layer(bpy.data.objects['Lamp_Back'], 0)
  set_layer(bpy.data.objects['Ground'], 0)

  # Set the render settings back to what they were
  render_args.filepath = old_filepath
  render_args.engine = old_engine
  render_args.use_antialiasing = old_use_antialiasing

  return object_colors


def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))


# I wonder if there's a better way to do this?
def delete_object(obj):
  """ Delete a specified blender object """
  for o in bpy.data.objects:
    o.select = False
  obj.select = True
  bpy.ops.object.delete()


def get_camera_coords(cam, pos):
  """
  For a specified point, get both the 3D coordinates and 2D pixel-space
  coordinates of the point from the perspective of the camera.

  Inputs:
  - cam: Camera object
  - pos: Vector giving 3D world-space position

  Returns a tuple of:
  - (px, py, pz): px and py give 2D image-space coordinates; pz gives depth
    in the range [-1, 1]
  """
  scene = bpy.context.scene
  x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
  scale = scene.render.resolution_percentage / 100.0
  w = int(scale * scene.render.resolution_x)
  h = int(scale * scene.render.resolution_y)
  px = int(round(x * w))
  py = int(round(h - y * h))
  return (px, py, z)


def set_layer(obj, layer_idx):
  """ Move an object to a particular layer """
  # Set the target layer to True first because an object must always be on
  # at least one layer.
  obj.layers[layer_idx] = True
  for i in range(len(obj.layers)):
    obj.layers[i] = (i == layer_idx)


def add_object(object_dir, name, scale, loc, theta=0):
  """
  Load an object from a file. We assume that in the directory object_dir, there
  is a file named "$name.blend" which contains a single object named "$name"
  that has unit size and is centered at the origin.

  - scale: scalar giving the size that the object should be in the scene
  - loc: tuple (x, y) giving the coordinates on the ground plane where the
    object should be placed.
  """
  # First figure out how many of this object are already in the scene so we can
  # give the new object a unique name
  count = 0
  for obj in bpy.data.objects:
    if obj.name.startswith(name):
      count += 1

  name_path = name
  filepath = os.path.join(object_dir, '%s.blend' % name_path)

  with bpy.data.libraries.load(filepath, link=True) as (data_from, data_to):
    name = [name for name in data_from.objects][0]

  filename = os.path.join(object_dir, '%s.blend' % name_path, 'Object', name)
  bpy.ops.wm.append(filename=filename)

  # Give it a new name to avoid conflicts
  new_name = '%s_%d' % (name, count)
  bpy.data.objects[name].name = new_name

  # Set the new object as active, then rotate, scale, and translate it
  x, y = loc
  bpy.context.scene.objects.active = bpy.data.objects[new_name]
  bpy.context.object.rotation_euler[2] = theta
  bpy.ops.transform.resize(value=(scale, scale, scale))
  bpy.ops.transform.translate(value=(x, y, scale))


def load_materials(material_dir):
  """
  Load materials from a directory. We assume that the directory contains .blend
  files with one material each. The file X.blend has a single NodeTree item named
  X; this NodeTree item must have a "Color" input that accepts an RGBA value.
  """
  for fn in os.listdir(material_dir):
    if not fn.endswith('.blend'): continue
    name = os.path.splitext(fn)[0]
    filepath = os.path.join(material_dir, fn, 'NodeTree', name)
    bpy.ops.wm.append(filename=filepath)


def add_material(name, **properties):
  """
  Create a new material and assign it to the active object. "name" should be the
  name of a material that has been previously loaded using load_materials.
  """
  # Attach the new material to the active object
  # Make sure it doesn't already have materials
  obj = bpy.context.active_object
  # check if obj has materials attribute
  for i in range(len(obj.data.materials)):
    obj.data.materials.pop()

  # Figure out how many materials are already in the scene
  mat_count = len(bpy.data.materials)

  # Create a new material; it is not attached to anything and
  # it will be called "Material"
  bpy.ops.material.new()

  # Get a reference to the material we just created and rename it;
  # then the next time we make a new material it will still be called
  # "Material" and we will still be able to look it up by name
  mat = bpy.data.materials['Material']
  mat.name = 'Material_%d' % mat_count
  
  assert len(obj.data.materials) == 0
  obj.data.materials.append(mat)
  # Find the output node of the new material
  output_node = None
  for n in mat.node_tree.nodes:
    if n.name == 'Material Output':
      output_node = n
      break

  # Add a new GroupNode to the node tree of the active material,
  # and copy the node tree from the preloaded node group to the
  # new group node. This copying seems to happen by-value, so
  # we can create multiple materials of the same type without them
  # clobbering each other
  group_node = mat.node_tree.nodes.new('ShaderNodeGroup')
  group_node.node_tree = bpy.data.node_groups[name]
  logger = properties.get('logger', None)
  # Find and set the "Color" input of the new group node
  for inp in group_node.inputs:
    if inp.name in properties:
      logger.info("color input: %s" % properties[inp.name])
      inp.default_value = properties[inp.name]

  # Wire the output of the new group node to the input of
  # the MaterialOutput node
  mat.node_tree.links.new(
      group_node.outputs['Shader'],
      output_node.inputs['Surface'],
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
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')

