# Probing: Object Binding Interpretability

This folder contains a minimal pipeline to probe object bindings by:
1) generating 2D scenes + per-object instance masks, then
2) extracting per-layer image-token activations from a VLM, then
3) pooling token embeddings into per-object embeddings using the masks, then
4) training linear probes for color / shape / conjunction.

## Quickstart

1) Generate a masked dataset (does **not** write into benchmark dataset folders):

```bash
python probing/00_generate_masked_scene_desc_dataset.py \
  --out_dir data/probing/scene_description_balanced_2d \
  --n_objects 10 --n_trials 100 --img_size 40 --seed 0
```

2) Extract image-token hidden states (Qwen support included):

```bash
python probing/10_extract_image_tokens.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --out_dir data/probing/scene_description_balanced_2d_out \
  --model qwen3_vl_30b \
  --prompt_path prompts/scene_description_2D_parse.txt \
  --layers 0,10,20
```

3) Pool per-object embeddings using masks:

```bash
python probing/20_build_object_embeddings.py \
  --dataset_dir data/probing/scene_description_balanced_2d \
  --tokens_dir data/probing/scene_description_balanced_2d_out/tokens \
  --out_dir data/probing/scene_description_balanced_2d_out
```

4) Train linear probes:

```bash
python probing/30_train_probes.py \
  --embeddings_path data/probing/scene_description_balanced_2d_out/embeddings.npz \
  --out_dir data/probing/scene_description_balanced_2d_out/probes \
  --test_split 0.2 --seed 0 --C 1.0
```

## Adding support for a new model

`probing/model_introspect.py` currently extracts image-token hidden states via an HF forward pass.

- Implement a `get_hf_components()` method on the relevant wrapper in `models/` returning `(hf_model, hf_processor)`.
- Extend `probing/model_introspect.py:extract_image_hidden_states()` to:
  - prepare model inputs with the processor,
  - run a forward pass with `output_hidden_states=True`,
  - identify which sequence positions correspond to image tokens (and return them).

