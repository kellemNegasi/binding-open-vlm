# Global Probing: Illusory Conjunctions

This document describes the global scene probing experiment used to test for
illusory conjunctions in VLM internal representations. The goal is to detect
whether a global scene embedding increasingly supports absent-but-implied
color/shape conjunctions as triplet structure rises.

## Method overview

We treat each scene as a single representation and train a multi-label linear
probe that predicts which (color, shape) pairs are present in the image. We
then measure false positives on two absent sets:
- **Implied-absent**: absent pairs supported by a binding-swap pattern.
- **Non-implied absent**: absent pairs with no swap support.

If binding interference exists, implied-absent false positives should rise
with triplet structure more than non-implied false positives.

## Processing pipeline

1) **Extract internal tokens**
   - Run a forward pass with `output_hidden_states=True`.
   - Save per-layer image patch tokens.
   - Also save a CLS-like token (the vision-start token) when present.

2) **Build global embeddings**
   - **Mean pooled**: average all image patch tokens per layer.
   - **CLS**: use the CLS-like token per layer (if available).

3) **Construct labels**
   - Enumerate all (color, shape) pairs from the global vocab.
   - Mark each pair as:
     - present,
     - implied-absent,
     - non-implied absent.

4) **Train multi-label probe**
   - One-vs-rest logistic regression (one classifier per pair).
   - Train/test split at the scene level.

5) **Evaluate**
   - Compute FPR on implied-absent vs non-implied absent pairs.
   - Group results by `triplet_count` and by `n_implied_absent_pairs`.

## Label definitions

Let the scene contain objects `(c_i, s_i)`.

- **Present**: `(c, s)` appears in the scene.
- **Implied-absent**: `(c, s)` is absent but:
  - `c` appears with a different shape, and
  - `s` appears with a different color.
- **Non-implied absent**: absent but not implied.

### Mathematical form (LaTeX)

Let the scene be a multiset of objects
\[
\mathcal{O} = \{(c_i, s_i)\}_{i=1}^N,
\]
with color set \(\mathcal{C}\) and shape set \(\mathcal{S}\).
Define the present set:
\[
\mathcal{P} = \{(c, s) \in \mathcal{C}\times\mathcal{S} : \exists i,\ (c_i, s_i) = (c, s)\}.
\]

An absent pair is any \((c, s) \in \mathcal{C}\times\mathcal{S} \setminus \mathcal{P}\).
An absent pair is **implied** if:
\[
(c, s) \notin \mathcal{P},
\quad \exists i:\ (c_i, s_i)=(c, s_i)\ \text{with}\ s_i \neq s,
\quad \exists j:\ (c_j, s_j)=(c_j, s)\ \text{with}\ c_j \neq c.
\]

Define the implied-absent set:
\[
\mathcal{I} = \{(c, s)\in \mathcal{C}\times\mathcal{S} \setminus \mathcal{P}:\ \text{conditions above hold}\}.
\]
The non-implied absent set is:
\[
\mathcal{N} = \big(\mathcal{C}\times\mathcal{S}\big) \setminus (\mathcal{P}\cup\mathcal{I}).
\]

#### Triplet count (object-triple definition)

Let \(\binom{\mathcal{O}}{3}\) be all unordered triples of objects.
Define the per-triple indicator:
\[
T(o_a, o_b, o_c) = \mathbb{1}\big(\text{at least two colors match}\big)\cdot
\mathbb{1}\big(\text{at least two shapes match}\big).
\]
Then the unique-triplet count is:
\[
\text{Triplets} = \sum_{(o_a, o_b, o_c)\in \binom{\mathcal{O}}{3}} T(o_a, o_b, o_c).
\]
Note: the dataset's `triplet_count` can count such structures with multiplicity
depending on the internal co-occurrence matrix definition.

#### Label vector

Let \(K = |\mathcal{C}|\cdot|\mathcal{S}|\). Define a fixed ordering of pairs
\(\{(c_k, s_k)\}_{k=1}^K\). The multi-label target for a scene is:
\[
\mathbf{y}\in\{0,1\}^K,\quad y_k = \mathbb{1}\big((c_k, s_k)\in \mathcal{P}\big).
\]
Implied-absent and non-implied masks are:
\[
\mathbf{m}^{\text{imp}}_k = \mathbb{1}\big((c_k, s_k)\in \mathcal{I}\big),\quad
\mathbf{m}^{\text{non}}_k = \mathbb{1}\big((c_k, s_k)\in \mathcal{N}\big).
\]

## Example (balanced 2D scene)

Example image:
`data/probing/scene_description_balanced_2d/images/sample_000383.png`

![Balanced 2D sample 000383](../data/probing/scene_description_balanced_2d/images/sample_000383.png)

Metadata excerpt (from `data/probing/scene_description_balanced_2d/meta.jsonl`):
```json
{
  "sample_id": 383,
  "image": "images/sample_000383.png",
  "triplet_count": 4,
  "n_objects": 10,
  "n_shapes": 2,
  "n_colors": 2,
  "objects": [
    {"color": "purple", "shape": "star"},
    {"color": "black", "shape": "right-arrow"},
    {"color": "black", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"},
    {"color": "black", "shape": "right-arrow"},
    {"color": "purple", "shape": "right-arrow"}
  ]
}
```

Global vocab:
- colors = {black, purple}
- shapes = {right-arrow, star}

Present pairs:
- (purple, star)
- (black, right-arrow)
- (purple, right-arrow)

Absent pairs:
- (black, star)

Implied-absent pairs:
- (black, star) is implied because:
  - black appears with right-arrow, and
  - star appears with purple.

Non-implied absent pairs:
- none (only one absent pair exists and it is implied).

## Running the pipeline

Scripts (defaults to Qwen3-30B):
```bash
bash scripts/extract_image_tokens.sh
bash scripts/global_embedding_labels.sh
bash scripts/train_multi_label_probe.sh
```

Model overrides:
```bash
bash scripts/extract_image_tokens.sh qwen2_5_vl_7b qwen2.5-VL-7B-Instruct
bash scripts/global_embedding_labels.sh qwen2.5-VL-7B-Instruct
bash scripts/train_multi_label_probe.sh qwen2.5-VL-7B-Instruct
```

Outputs:
- `global_embeddings.npz` with `X_{layer}` (mean) and `X_cls_{layer}` (CLS).
- `probe_results.json` with FPR by triplet count and implied-absent count.
