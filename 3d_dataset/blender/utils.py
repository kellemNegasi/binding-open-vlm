import os
import sys
import bpy
import bpy_extras


def extract_args(input_argv=None):
    """
    Pull out command-line arguments after "--".
    """
    if input_argv is None:
        input_argv = sys.argv
    if "--" not in input_argv:
        return []
    idx = input_argv.index("--")
    return input_argv[idx + 1 :]


def delete_object(obj):
    """
    Delete a specified blender object (Blender 2.80+ / 5.x selection API).
    """
    if obj is None:
        return
    # Deselect all
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.delete()


def get_camera_coords(cam, pos):
    """
    For a specified point, get both the 3D coordinates and 2D pixel-space
    coordinates from camera view.
    """
    scene = bpy.context.scene
    x, y, z = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    scale = scene.render.resolution_percentage / 100.0
    w = int(scale * scene.render.resolution_x)
    h = int(scale * scene.render.resolution_y)
    px = int(round(x * w))
    py = int(round(h - y * h))
    return (px, py, z)


def add_object(object_dir, name, scale, loc, theta=0.0):
    """
    Append an object named `name` from `<object_dir>/<name>.blend` and place it.

    Assumption: `<name>.blend` contains an Object datablock called exactly `name`.
    """
    object_dir = os.path.abspath(object_dir)

    # Unique naming
    count = 0
    for obj in bpy.data.objects:
        if obj.name.startswith(name):
            count += 1

    # Append object datablock
    directory = os.path.join(object_dir, f"{name}.blend", "Object")
    bpy.ops.wm.append(directory=directory, filename=name)

    # Rename appended object to avoid collisions
    new_name = f"{name}_{count}"
    if name not in bpy.data.objects:
        raise RuntimeError(
            f"Failed to append object '{name}' from {directory}. "
            f"Check that the .blend exists and contains Object '{name}'."
        )
    bpy.data.objects[name].name = new_name

    obj = bpy.data.objects[new_name]

    # Make active & selected
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Apply transforms directly (robust in background mode)
    x, y = loc
    obj.rotation_euler[2] = float(theta)
    obj.scale = (float(scale), float(scale), float(scale))
    obj.location = (float(x), float(y), float(scale))


def load_materials(material_dir):
    """
    Append NodeTree groups from .blend files in material_dir.
    Each X.blend should contain a NodeTree item named X.
    """
    material_dir = os.path.abspath(material_dir)
    for fn in os.listdir(material_dir):
        if not fn.endswith(".blend"):
            continue
        name = os.path.splitext(fn)[0]
        directory = os.path.join(material_dir, fn, "NodeTree")
        bpy.ops.wm.append(directory=directory, filename=name)


def add_material(name, **properties):
    """
    Create a new material and assign it to the active object.
    `name` must match a node group loaded into bpy.data.node_groups.
    The node group should expose inputs like "Color".
    """
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("No active object to assign material to.")

    # Create material
    mat = bpy.data.materials.new(name=f"{name}_inst")
    mat.use_nodes = True

    # Clear default nodes
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    group = nt.nodes.new("ShaderNodeGroup")

    if name not in bpy.data.node_groups:
        raise RuntimeError(f"Material node group '{name}' not found. Did you call load_materials()?")

    group.node_tree = bpy.data.node_groups[name]

    # Set provided inputs (e.g., Color=rgba)
    for inp in group.inputs:
        if inp.name in properties:
            inp.default_value = properties[inp.name]

    # Link group shader to output surface
    if "Shader" in group.outputs:
        nt.links.new(group.outputs["Shader"], out.inputs["Surface"])
    else:
        # Fallback: common alternative output name
        # (Some nodegroups might output BSDF/Surface)
        for k in group.outputs.keys():
            nt.links.new(group.outputs[k], out.inputs["Surface"])
            break

    # Assign to object
    obj.data.materials.clear()
    obj.data.materials.append(mat)
