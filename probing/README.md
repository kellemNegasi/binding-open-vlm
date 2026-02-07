# Probing: Object Binding Interpretability

This folder contains two related probing pipelines:

1) **Object-level probes**: pool *per-object* embeddings using instance masks, then train linear probes for
   color / shape / (color,shape) conjunction.
2) **Global scene probe (illusory conjunctions)**: pool *per-scene* embeddings, then train a multi-label probe
   over all (color,shape) pairs and report false positives on implied vs non-implied absent pairs.

## Quickstart

### Object-level probing (masked objects → pooled embeddings → probes)

1) Generate a masked dataset (does **not** write into benchmark dataset folders):

```bash
python probing/generate_masked_scene_desc_dataset.py \
  --out_dir data/probing/scene_description_balanced_2d \
  --n_objects 10 --n_trials 100 --img_size 40 --seed 0
```

2) Extract image-token hidden states (Qwen support included):

```bash
python probing/extract_image_tokens.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --out_dir data/probing/scene_description_balanced_2d_out \
  --model qwen3_vl_30b \
  --prompt_path prompts/scene_description_2D_parse.txt \
  --layers 0,10,20

# To extract *all* transformer blocks:
#   --layers all
```

### Optional: debug image-token positions

To sanity-check that we are extracting hidden states from *image patch tokens* (not the text tokens), run:

```bash
python probing/debug_token_positions.py \
  --model qwen3_vl_30b \
  --prompt_path prompts/scene_description_2D_parse.txt \
  --dataset_dir data/probing/scene_description_balanced_2d
```

### Running on XPU (Qwen2.5-VL-7B)

If your local PyTorch build supports `torch.xpu` (e.g., via Intel Extension for PyTorch), you can request XPU:

```bash
python probing/extract_image_tokens.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --out_dir data/probing/scene_description_balanced_2d_out_qwen25_xpu \
  --model qwen2.5-VL-7B-Instruct \
  --device xpu --dtype bf16 \
  --layers 0,10,20
```

3) Pool per-object embeddings using masks:

```bash
python probing/build_object_embeddings.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --tokens_dir data/probing/scene_description_balanced_2d_out/tokens \
  --out_dir data/probing/scene_description_balanced_2d_out
```

4) Train linear probes:

```bash
python probing/train_probes.py \
  --embeddings_path data/probing/scene_description_balanced_2d_out/embeddings.npz \
  --out_dir data/probing/scene_description_balanced_2d_out/probes \
  --test_split 0.2 --seed 0 --C 1.0
```

## Global scene probe (illusory conjunctions)

This variant builds a single embedding per scene (mean-pooled image tokens and optional CLS token),
trains a multi-label probe to predict which (color, shape) pairs are present, and evaluates false positives
on implied vs non-implied absent pairs. Results are grouped by both triplet count and implied-absent count.

Key definitions:
- **Pair index**: each label corresponds to one (color, shape) pair in the global vocab.
- **Present**: the pair appears in the scene.
- **Implied-absent**: the pair is absent, but the color appears with another shape and the shape appears with another color.
- **Non-implied absent**: absent and not implied.
- **FPR**: for an absent set, the fraction of pairs predicted as present.

### Recommended: use the sbatch jobs (cluster)

- `probing_global_job.sbatch`: balanced 2D dataset, runs token extraction → global embeddings/labels → global probe training.
- `probing_global_job_limited.sbatch`: limited-vocab balanced 2D dataset (max 4 colors × 4 shapes), same pipeline.

Both jobs call the same underlying scripts in `scripts/`:
- `scripts/extract_image_tokens.sh` → `probing/extract_image_tokens.py`
- `scripts/global_embedding_labels.sh` → `probing/build_global_embeddings.py`
- `scripts/train_multi_label_probe_per_sample.sh` → `probing/train_global_probe.py`

### Recommended: use the scripts (local or interactive)

```bash
bash scripts/extract_image_tokens.sh
bash scripts/global_embedding_labels.sh
bash scripts/train_multi_label_probe_per_sample.sh
```

Override the model or output paths as needed:

```bash
bash scripts/extract_image_tokens.sh qwen2_5_vl_7b qwen2.5-VL-7B-Instruct
bash scripts/global_embedding_labels.sh qwen2.5-VL-7B-Instruct
bash scripts/train_multi_label_probe_per_sample.sh qwen2.5-VL-7B-Instruct
```

Notes:
- `scripts/train_multi_label_probe.sh` runs a single seed; `scripts/train_multi_label_probe_per_sample.sh` supports
  multiple seeds and saves per-sample intermediates (this is what `probing_global_job.sbatch` uses).
- You can control parallelism via env vars: `OVR_N_JOBS` (One-vs-Rest over labels) and `FEATURE_JOBS` (across features).

### Direct commands (manual)

1) Extract image tokens (saves image tokens + CLS tokens when available):

```bash
python probing/extract_image_tokens.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --out_dir data/probing/scene_description_balanced_2d_out_qwen3-vl-30b-a3b-instruct \
  --model qwen3_vl_30b \
  --prompt_path prompts/scene_description_2D_parse.txt \
  --layers 0,10,20 --overwrite
```

2) Build global scene embeddings + labels:

```bash
python probing/build_global_embeddings.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --tokens_dir data/probing/scene_description_balanced_2d_out_qwen3-vl-30b-a3b-instruct/tokens \
  --out_dir output/probing_global/scene_description_balanced_2d_qwen3-vl-30b-a3b-instruct \
  --pool mean
```

3) Train the multi-label probe and report FPRs:

```bash
python probing/train_global_probe.py \
  --embeddings_path output/probing_global/scene_description_balanced_2d_qwen3-vl-30b-a3b-instruct/global_embeddings.npz \
  --out_dir output/probing_global/scene_description_balanced_2d_qwen3-vl-30b-a3b-instruct/probes \
  --test_split 0.2 --seed 0 --C 1.0 --threshold 0.5
```

Outputs:
- `global_embeddings.npz` contains `X_{layer}` (mean pooled) and `X_cls_{layer}` (CLS) features.
- `probe_results.json` reports micro/macro F1 and implied/non-implied FPRs, grouped by:
  - `triplet_count_per_sample`
  - `n_implied_absent_pairs_per_sample`
- `fpr_by_layer_triplet.csv` stores implied/non-implied FPR per (layer, triplet_count) pair.
- `macro_fpr_by_layer.csv` stores macro/max FPR over triplet_count per layer.
- Per-seed intermediates are saved under `probes/intermediates/` (when enabled) for any custom post-hoc aggregation.

### Train/test splitting note

By default, `probing/train_probes.py` splits by `sample_id` (image), meaning all objects from the same image stay in the same split.
This avoids leakage where objects from the same rendered scene appear in both train and test.

If you explicitly want to split over *objects* (older behavior), you can use:

```bash
python probing/train_probes.py \
  --embeddings_path data/probing/scene_description_balanced_2d_out/embeddings.npz \
  --out_dir data/probing/scene_description_balanced_2d_out/probes \
  --split_mode object
```

## Which module does what (in `./probing`)

- `probing/generate_masked_scene_desc_dataset.py`: renders 2D scenes and writes `meta.jsonl`, RGB images, and per-object masks.
- `probing/extract_image_tokens.py`: loads a VLM via Hydra and saves per-layer image-token hidden states to `tokens/*.npz`.
- `probing/model_introspect.py`: model/processor loading and the logic for locating image-token positions + extracting hidden states.
- `probing/build_object_embeddings.py`: pools token embeddings into per-object embeddings using masks/boxes → writes `embeddings.npz`.
- `probing/train_probes.py`: trains simple linear probes (color/shape/conjunction) on per-object embeddings.
- `probing/build_global_embeddings.py`: mean-pools image tokens per scene and builds multi-label targets for all (color,shape) pairs.
- `probing/train_global_probe.py`: trains a multi-label global probe and reports F1 + implied/non-implied absent FPR metrics.
- `probing/debug_token_positions.py`: sanity-check utility to print where image tokens fall in the model’s input sequence.

## What’s currently used vs older artifacts

There are two “active” entrypoint sets in this repo:

- **Global scene probing (illusory conjunctions)** uses:
  - `probing/extract_image_tokens.py`, `probing/model_introspect.py`, `probing/build_global_embeddings.py`, `probing/train_global_probe.py`
  - Driven by `probing_global_job.sbatch` (and `scripts/*` above).

- **Object-level probing** (masked dataset → pooled object embeddings → probes) uses:
  - `probing/generate_masked_scene_desc_dataset.py`, `probing/extract_image_tokens.py`, `probing/model_introspect.py`,
    `probing/build_object_embeddings.py`, `probing/train_probes.py`
  - Driven by `probing_job.sbatch` (or the manual commands in the quickstart).

Modules that are not part of the “mainline” runs but are still useful:
- `probing/debug_token_positions.py` (debug helper; referenced in `probing_job.sbatch` but commented out by default).

## Adding support for a new model

`probing/model_introspect.py` currently extracts image-token hidden states via an HF forward pass.

- Implement a `get_hf_components()` method on the relevant wrapper in `models/` returning `(hf_model, hf_processor)`.
- Extend `probing/model_introspect.py:extract_image_hidden_states()` to:
  - prepare model inputs with the processor,
  - run a forward pass with `output_hidden_states=True`,
  - identify which sequence positions correspond to image tokens (and return them).
