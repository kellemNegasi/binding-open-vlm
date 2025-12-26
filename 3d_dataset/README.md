# 3D Dataset Generator

This module adapts the CLEVR Blender tooling so we can reproduce the 3D datasets
described in the *Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem*
paper. It reuses CLEVR's base scene, shapes, and materials, but layers on
task-specific samplers so that the rendered images match the sampling regimes in
the paper (visual search, counting with entropy manipulations, and scene
descriptions with controlled feature-triplet counts).

## Layout

```
3d_dataset/
├── README.md
├── __init__.py
├── devices.py              # Device detection + Cycles helper
├── sampling.py             # Task-specific object samplers
├── gen_blender.py          # Entry-point that orchestrates sampling + Blender runs
└── blender/
    ├── data/               # CLEVR base scene/materials plus added shapes/colors
    ├── render_binding.py   # Blender script that consumes explicit scene specs
    └── utils.py            # CLEVR helper functions for interacting with Blender
```

The `blender/data` folder contains the CLEVR base scene/materials plus an expanded mesh/color kit.
It includes the full 3D scene-description prompt vocabulary used in `binding/prompts/scene_description_3D.txt`,
so those prompts are fully supported (see “Matching the paper’s 3D asset list”).
You can drop in more `.blend` files later—just add them to `data/shapes/` and register them inside
`data/properties.json`.

## Requirements

* Blender 2.9+ with the Cycles renderer enabled (`blender` on `$PATH`, or pass `--blender-binary`).
  No Blender Python-path setup is required: `render_binding.py` adds its own directory to `sys.path`.
* Python dependencies that are already part of this repo (`pandas` is used to emit metadata).

## Generating a dataset

All datasets are generated through `gen_blender.py`. The script samples object
configurations, writes metadata to `data/vlm/3D/<task>/metadata.csv`, and invokes Blender
to render the PNGs into `data/vlm/3D/<task>/images/`.

Examples:

```bash
# Generate all 3D datasets (search + counting + scene description)
./3d_dataset/generate_3d_all.sh

# Quick preview render (fewer scenes + low samples)
./3d_dataset/generate_3d_all.sh --num-scenes 20 --scene-num-scenes 20 --preview

# Parallelize generation across N shards (runs shards locally in the background)
./3d_dataset/generate_3d_all.sh --jobs 4

# HPC job-array style: run only shard K of N (then merge metadata after all shards finish)
./3d_dataset/generate_3d_all.sh --jobs 8 --job-idx 3 --only disjunctive_search
./3d_dataset/merge_3d_metadata.sh --jobs 8 disjunctive_search

# Red-vs-green popout dataset (1000 scenes with 4–50 distractors)
python -m 3d_dataset.gen_blender \
  --task disjunctive_search \
  --num-scenes 1000 \
  --root-dir /abs/path/to/binding \
  --data-dir /abs/path/to/binding/data

# Conjunctive search (cubes vs spheres)
python -m 3d_dataset.gen_blender --task conjunctive_search

# Counting with specified entropy level (low/medium_color/medium_shape/high)
python -m 3d_dataset.gen_blender --task counting_high_entropy --num-scenes 1200

# Scene description benchmark w/ custom triplet histogram
python -m 3d_dataset.gen_blender \
  --task scene_description \
  --triplet-targets "0:150,1:200,2:200,3:200,4:200,5:150"
```

Counting entropy definitions match the paper exactly (same-vs-unique shape/color constraints). For
`medium_*`/`high_entropy`, ensure your `properties.json` has at least as many distinct shapes/colors as
your requested maximum count (or lower `--count-max`).

Common flags:

| Flag | Purpose |
| --- | --- |
| `--root-dir` | Absolute path to the repo root (used to keep metadata paths relative). |
| `--data-dir` | Where rendered assets + metadata should live. Defaults to `<root>/data`. |
| `--device` | Preferred compute backend (`auto`, `cpu`, `gpu`, `xpu`, etc.). We auto-detect CUDA, HIP, ONEAPI, or fall back to CPU. |
| `--blender-binary` | Override if Blender is not on your `$PATH`. |
| `--save-blendfiles` | Persist the per-scene `.blend` files for inspection/debugging. |
| `--distractor-min/--distractor-max` | Control the number of distractors for the search datasets. |
| `--count-min/--count-max` | Control the counts for the numerosity datasets. |
| `--triplet-targets` | Histogram for scene-description triplets (`triplets:count,...`). |
| `--scene-min-objects/--scene-max-objects` | Clamp object counts when sampling scene-description tasks. |
| `--width/--height` | Output resolution in pixels (e.g., lower to 256×256 for fast smoke tests, raise to 512+ for final data). |
| `--render-samples` | Cycles sample count; higher values reduce noise but increase render time (64 for previews, 256+ for production). |
| `--min-pixels-per-object` | Passed to Blender’s visibility check. Lower it for tiny preview renders so layouts stop retrying forever. |
| `--max-retries` | Per-object placement retries inside Blender before a whole-scene restart (raise for 50-object search scenes). |
| `--max-layout-attempts` | Caps whole-scene placement retries; if exceeded we render whatever objects were successfully placed. |
| `--num-shards/--shard-index` | Deterministically split a task across multiple jobs; shard selection is based on the sampled scene index. |
| `--metadata-file` | Use a per-shard filename (e.g., `metadata.part-0.csv`) to avoid collisions when running in parallel. |
| `--combined-scene-file` | Output filename for the combined scene JSON; set per-shard to avoid collisions when running in parallel. |

All rendered metadata includes a `path` column (relative to `root_dir`) so the
existing Hydra tasks can consume the CSVs without modification. Additional task-
specific columns (e.g., `popout`, `condition`, `triplet_count`) are populated to
mirror the columns exposed by the existing 2D datasets.

For quick smoke tests, shrink the resolution (`--width/--height 256`), samples (`--render-samples 32`),
and visibility requirement (`--min-pixels-per-object 10`) so a single scene renders in a few seconds.
Once the pipeline looks good, bump those knobs back up for production-quality datasets.

## Rendering devices

`gen_blender.py` tries to detect the best available Cycles backend:

* CUDA (`nvidia-smi` present) → `--compute_device_type CUDA`
* HIP (`rocminfo` present) → `--compute_device_type HIP`
* ONEAPI (`sycl-ls`/`ONEAPI_DEVICE_SELECTOR`) → `--compute_device_type ONEAPI`
* macOS → `--compute_device_type METAL`
* Otherwise the renderer stays on CPU.

You can override this detection with `--device cpu|cuda|hip|oneapi|metal`.

## Adding new shapes/materials/colors

1. Drop the `.blend` file for the new shape into `blender/data/shapes/<ShapeName>.blend`,
   ensuring the object is centered at the origin with unit size. For best results, keep a single mesh
   object in the file and name it the same as the filename stem (the loader will auto-detect if they
   differ). If the file contains multiple mesh objects, the loader will automatically join them into a
   single mesh so downstream visibility/material logic works reliably.
2. Update `blender/data/properties.json`:
   * Add an entry under `"shapes"` mapping the logical name to the `.blend` stem.
   * Add any new colors (RGB values) or materials (node-group file names).
3. Re-run `gen_blender.py`. The samplers pull the vocabulary directly from
   `properties.json`, so the new assets are immediately available (and required
   combinations such as the spikey ball used in the scene-description prompts can
   be added once their meshes exist).

## Matching the paper’s 3D asset list

The 3D prompts in Appendix A of the paper (see `prompts/scene_description_3D.txt`) enumerate the shapes
and colors annotators expect: `cone, cylinder, bowl, donut, sphere, cube, droplet, bowling-pin, coil,
crown, snowman, spikey-ball` with colors drawn from
`red, green, blue, yellow, purple, light green, gray, black, light blue, pink, teal, brown`.
This repo now includes meshes and a `properties.json` vocabulary that match those prompt tokens.
Note that `spikey-ball` is mapped to the on-disk blendfile `spike-ball.blend`, and the Blender
Object datablock name is auto-detected at append time (so it does not need to match the filename).

### Asset checklist

| Asset type | Needed for prompts | Present today | Missing items |
| --- | --- | --- | --- |
| Shapes (`blender/data/shapes/*.blend`) | 12 | 12 (`cone`, `cylinder`, `bowl`, `donut`, `sphere`, `cube`, `droplet`, `bowling-pin`, `coil`, `crown`, `snowman`, `spikey-ball`) | — |
| Colors (`properties.json` → `"colors"`) | 12 | 12 (`red`, `green`, `blue`, `yellow`, `purple`, `light green`, `gray`, `black`, `light blue`, `pink`, `teal`, `brown`) | — |

### Expanding the mesh/color vocabulary (optional)

If you want to run stricter counting settings (e.g., high-entropy scenes up to 20 objects requiring
unique shapes/colors) you may want a larger vocabulary than the paper’s scene-description prompt list.
The quickest path is to stay inside Blender, create each mesh as a separate object, apply all modifiers,
shade smooth, and save the `.blend` into `blender/data/shapes/<ShapeName>.blend`.

1. **Cone** – Add → Mesh → Cone, set the radius so it visually matches CLEVR scales (radius ≈ 1), apply
   rotation/scale, bevel the base slightly, shade smooth.
2. **Bowl** – Add → Mesh → UV Sphere, delete the top half, add a Solidify modifier for thickness, bevel
   the rim, apply modifiers.
3. **Donut (torus)** – Add → Mesh → Torus, set major/minor radii to resemble the paper’s donut, apply
   scale and shade smooth.
4. **Droplet** – Start with a UV sphere, use proportional editing to pull the top vertex upward into a
   teardrop, optionally add a Subdivision Surface modifier, then apply.
5. **Bowling-pin** – Create a Bezier curve profile (side view), convert to mesh via Screw modifier around
   the Z axis, then apply and shade smooth.
6. **Coil** – Add → Curve → Path, apply a Screw modifier with a small angle to make a spiral, convert to
   mesh, and bevel to round out the wire.
7. **Crown** – Begin with a cylinder, delete alternating top faces, extrude upwards into points, bevel
   sharp edges.
8. **Snowman** – Stack two or three spheres of decreasing radius, join them into a single mesh, and use
   the Boolean Union modifier to remove seams.
9. **Spikey-ball** – Either (a) add an Icosphere, select all faces, and Extrude Along Normals for spikes,
   or (b) import a public-domain STL, then `File → External Data → Unpack` so the `.blend` stays
   self-contained.

Store each finished mesh as `blender/data/shapes/<LogicalName>.blend` and add the logical-to-file mapping
under `"shapes"` inside `properties.json`. Re-run `gen_blender.py` once the vocabulary includes the new
entries so they propagate to `SceneVocabulary`.

### Adding the missing colors

Edit `blender/data/properties.json` and append entries such as:

```json
"light green": [144, 238, 144],
"black": [0, 0, 0],
"light blue": [173, 216, 230],
"pink": [255, 105, 180],
"teal": [41, 208, 208]
```

Use RGB values in the 0–255 range to stay consistent with the existing palette. After saving the file,
re-run any generation script so `SceneVocabulary.from_properties` picks up the expanded color list.

### Verifying the asset kit

1. Run `python -m 3d_dataset.gen_blender --task scene_description --num-scenes 1 --save-blendfiles`
   and inspect the sampled metadata; each promised shape/color should now appear at least once after
   enough renders.
2. If Blender reports “Shape '<name>' missing from properties.json”, double-check the spelling in both
   `SceneBlueprint` metadata (`prompts/*.txt`) and `properties.json`.
3. Commit the new `.blend` files along with the updated properties so future runs faithfully reproduce
   the paper’s object vocabulary.

## Integration with `Task`

The generator writes metadata to the same folder structure that `tasks.task.Task`
expects (`data/vlm/<variant>/<task>/metadata.csv`). Once the CSV and image folder
are in place, setting `task_variant=3D` in your Hydra config is enough for the task
loader to pick up the pre-rendered assets without attempting to regenerate them in Python.
