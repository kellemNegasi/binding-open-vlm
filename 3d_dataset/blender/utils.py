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
    # Prefer datablock removal (works in background mode and avoids selection issues).
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
        return
    except ReferenceError:
        # Object already removed.
        return
    except Exception:
        pass

    # Fallback to operator-based deletion if datablock removal failed.
    try:
        for o in bpy.context.view_layer.objects:
            try:
                o.select_set(False)
            except ReferenceError:
                continue
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.delete()
    except ReferenceError:
        return


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

    This function is resilient to mismatches between the `.blend` filename and the
    internal Object datablock name: it inspects the `.blend` library contents and
    appends a best-match object (preferring an object whose name matches the
    filename stem, otherwise falling back to the first available object).
    """
    object_dir = os.path.abspath(object_dir)

    blend_path = os.path.join(object_dir, f"{name}.blend")
    if not os.path.exists(blend_path):
        raise RuntimeError(f"Shape blendfile not found: {blend_path}")

    existing_names = {obj.name for obj in bpy.data.objects}
    prefix_count = sum(1 for obj_name in existing_names if obj_name.startswith(name))

    with bpy.data.libraries.load(blend_path, link=False) as (data_from, _):
        library_objects = list(getattr(data_from, "objects", []) or [])

    if not library_objects:
        raise RuntimeError(f"No Object datablocks found in {blend_path}")

    def _is_background_name(obj_name: str) -> bool:
        n = (obj_name or "").strip().lower()
        if not n:
            return True
        if n in {"camera", "light", "lamp"}:
            return True
        if n.startswith(("camera", "light", "lamp")):
            return True
        if n in {"plane", "ground", "floor", "backdrop"}:
            return True
        if n.startswith(("plane", "ground", "floor", "backdrop")):
            return True
        return False

    def _is_cutter_name(obj_name: str) -> bool:
        n = (obj_name or "").lower()
        return ("_cut" in n) or ("cut_" in n) or ("cutter" in n)

    preferred = []
    if name in library_objects:
        preferred.append(name)
    else:
        name_lower = name.lower()
        for candidate in library_objects:
            if candidate.lower() == name_lower:
                preferred.append(candidate)
                break
    # If we can't find a name match, appending a single arbitrary object is almost
    # always wrong for multi-part assets (e.g. "snowman" might be built from many
    # meshes named Sphere/Cone/etc). In that case, fall back to appending all
    # non-background objects and joining the meshes.
    append_all = not preferred
    if append_all:
        candidates = [
            obj_name
            for obj_name in library_objects
            if not _is_background_name(obj_name)
        ]
        if not candidates:
            candidates = list(library_objects)
    else:
        candidates = preferred + [obj_name for obj_name in library_objects if obj_name not in preferred]

    directory = os.path.join(blend_path, "Object")
    new_name = f"{name}_{prefix_count}"

    def remove_object(o):
        try:
            bpy.data.objects.remove(o, do_unlink=True)
        except Exception:
            try:
                delete_object(o)
            except Exception:
                pass

    obj = None
    created_objs = []
    last_error = None
    for idx_candidate, desired_object in enumerate(candidates):
        before_names = {o.name for o in bpy.data.objects}
        try:
            if append_all:
                # In append-all mode, append every candidate (each is an Object datablock).
                # We do it in the first iteration and break out afterwards.
                if idx_candidate != 0:
                    continue
                for obj_name in candidates:
                    bpy.ops.wm.append(directory=directory, filename=obj_name)
            else:
                bpy.ops.wm.append(directory=directory, filename=desired_object)
        except Exception as e:
            last_error = e
            continue

        new_objs = [o for o in bpy.data.objects if o.name not in before_names]
        created_objs = list(new_objs)
        mesh_objs = [o for o in new_objs if getattr(o, "type", None) == "MESH" and getattr(o, "data", None)]
        if not mesh_objs:
            # Collect debug info before removing objects, otherwise accessing them can
            # trigger ReferenceError ("StructRNA ... has been removed").
            new_object_types = []
            for o in new_objs:
                try:
                    new_object_types.append(getattr(o, "type", None))
                except ReferenceError:
                    new_object_types.append(None)
            for o in new_objs:
                remove_object(o)
            last_error = RuntimeError(
                f"Appended '{desired_object}' but no mesh objects were introduced; "
                f"new object types were {new_object_types}"
            )
            continue

        # Decide which meshes should be treated as the logical object. When we appended
        # a single named object, extra meshes are often boolean cutters/helpers; when we
        # appended everything (no name match), we intentionally join all non-helper meshes.
        join_targets = []
        if append_all:
            for m in mesh_objs:
                if _is_background_name(getattr(m, "name", "")) or _is_cutter_name(getattr(m, "name", "")):
                    continue
                join_targets.append(m)
        else:
            non_helper_meshes = []
            for m in mesh_objs:
                n = getattr(m, "name", "")
                if _is_background_name(n) or _is_cutter_name(n):
                    continue
                non_helper_meshes.append(m)

            # Prefer the explicitly requested mesh if it exists; otherwise choose the
            # most-detailed mesh as the root. If there are multiple non-helper meshes,
            # join them so the logical object is complete.
            root = None
            for m in non_helper_meshes:
                try:
                    if getattr(m, "name", "").lower() == str(desired_object).lower():
                        root = m
                        break
                except ReferenceError:
                    continue

            if root is None:
                candidates_for_root = non_helper_meshes or mesh_objs
                def _poly_count(o):
                    try:
                        return len(getattr(getattr(o, "data", None), "polygons", []) or [])
                    except Exception:
                        return 0
                root = max(candidates_for_root, key=_poly_count)

            join_targets = list(non_helper_meshes) if len(non_helper_meshes) > 1 else [root]

            # Hide obvious cutters/background meshes to avoid rendering them, but keep them
            # in the scene so modifiers continue to work.
            for m in mesh_objs:
                if m in join_targets:
                    continue
                try:
                    if _is_cutter_name(getattr(m, "name", "")) or _is_background_name(getattr(m, "name", "")):
                        m.hide_render = True
                        m.hide_viewport = True
                except Exception:
                    pass

        # If we have multiple targets (append_all), join them into one mesh.
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

        if len(join_targets) > 1:
            for o in bpy.context.view_layer.objects:
                o.select_set(False)
            for m in join_targets:
                m.select_set(True)
            bpy.context.view_layer.objects.active = join_targets[0]
            try:
                bpy.ops.object.join()
            except Exception as e:
                last_error = e
                for o in new_objs:
                    remove_object(o)
                continue

        obj = bpy.context.view_layer.objects.active
        if obj is None or getattr(obj, "type", None) != "MESH" or not getattr(obj, "data", None):
            obj = join_targets[0]
            bpy.context.view_layer.objects.active = obj

        obj.name = new_name
        break

    if obj is None:
        raise RuntimeError(
            f"Failed to append a mesh object from {directory}. "
            f"Tried candidates={candidates}. Last error={last_error}"
        )

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

    # Return the logical root object plus every datablock we appended, so callers can
    # clean up everything (including hidden helpers) on retry.
    return obj, created_objs


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
