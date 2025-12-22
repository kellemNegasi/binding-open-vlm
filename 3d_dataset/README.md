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
    ├── data/               # Copied CLEVR assets (base scene, shapes, materials, properties)
    ├── render_binding.py   # Blender script that consumes explicit scene specs
    └── utils.py            # CLEVR helper functions for interacting with Blender
```

The `blender/data` folder currently contains the CLEVR sphere/cube/cylinder objects.
You can drop in new `.blend` files (e.g., spikey balls) later—just add them to
`data/shapes/` and register them inside `data/properties.json`.

## Requirements

* Blender 2.9+ with the Cycles renderer enabled. Add `binding/3d_dataset/blender`
  to Blender's Python path (e.g., via a `.pth` file) so the script can `import utils`.
* Python dependencies that are already part of this repo (`pandas` is used to emit metadata).

## Generating a dataset

All datasets are generated through `gen_blender.py`. The script samples object
configurations, writes metadata to `data/vlm/3D/<task>/metadata.csv`, and invokes Blender
to render the PNGs into `data/vlm/3D/<task>/images/`.

Examples:

```bash
# Red-vs-green popout dataset (1000 scenes with 4–50 distractors)
python 3d_dataset/gen_blender.py \
  --task disjunctive_search \
  --num-scenes 1000 \
  --root-dir /abs/path/to/binding \
  --data-dir /abs/path/to/binding/data

# Conjunctive search (cubes vs spheres)
python 3d_dataset/gen_blender.py --task conjunctive_search

# Counting with specified entropy level (low/medium_color/medium_shape/high)
python 3d_dataset/gen_blender.py --task counting_high_entropy --num-scenes 1200

# Scene description benchmark w/ custom triplet histogram
python 3d_dataset/gen_blender.py \
  --task scene_description \
  --triplet-targets "0:150,1:200,2:200,3:200,4:200,5:150"
```

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

All rendered metadata includes a `path` column (relative to `root_dir`) so the
existing Hydra tasks can consume the CSVs without modification. Additional task-
specific columns (e.g., `popout`, `condition`, `triplet_count`) are populated to
mirror the columns exposed by the existing 2D datasets.

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
   ensuring the object is centered at the origin with unit size.
2. Update `blender/data/properties.json`:
   * Add an entry under `"shapes"` mapping the logical name to the `.blend` stem.
   * Add any new colors (RGB values) or materials (node-group file names).
3. Re-run `gen_blender.py`. The samplers pull the vocabulary directly from
   `properties.json`, so the new assets are immediately available (and required
   combinations such as the spikey ball used in the scene-description prompts can
   be added once their meshes exist).

## Integration with `Task`

The generator writes metadata to the same folder structure that `tasks.task.Task`
expects (`data/vlm/<variant>/<task>/metadata.csv`). Once the CSV and image folder
are in place, setting `task_variant=3D` in your Hydra config is enough for the task
loader to pick up the pre-rendered assets without attempting to regenerate them in Python.
