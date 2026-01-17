import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _infer_grid(n_tokens: int, h_patches: int, w_patches: int) -> tuple[int, int]:
    if h_patches > 0 and w_patches > 0:
        return int(h_patches), int(w_patches)
    side = int(math.isqrt(n_tokens))
    if side * side != n_tokens:
        raise ValueError(
            f"n_image_tokens={n_tokens} is not a perfect square and no grid was provided. "
            "Re-run extraction with a model/processor that exposes H_patches/W_patches, "
            "or extend `probing/model_introspect.py`."
        )
    return side, side


def _patch_centers(h_patches: int, w_patches: int, image_size: int = 256) -> np.ndarray:
    ys = (np.arange(h_patches) + 0.5) * (image_size / h_patches)
    xs = (np.arange(w_patches) + 0.5) * (image_size / w_patches)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    centers = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)  # [n_tokens, 2] (x,y)
    return centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-object pooled embeddings from image-token activations.")
    parser.add_argument("--dataset_dir", type=str, default="data/probing/scene_description_balanced_2d")
    parser.add_argument("--tokens_dir", type=str, default="data/probing/scene_description_balanced_2d_out/tokens")
    parser.add_argument("--out_dir", type=str, default="data/probing/scene_description_balanced_2d_out")
    parser.add_argument("--layers", type=str, default=None, help="Optional comma-separated layer list; inferred if omitted.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / "meta.jsonl"
    records = _load_jsonl(meta_path)

    tokens_dir = Path(args.tokens_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Infer layer keys from the first available token file.
    first_npz = None
    for rec in records:
        cand = tokens_dir / f"sample_{int(rec['sample_id']):06d}.npz"
        if cand.exists():
            first_npz = cand
            break
    if first_npz is None:
        raise FileNotFoundError(f"No token files found under {tokens_dir}")

    with np.load(first_npz) as z:
        available_layers = sorted(int(k.split("_", 1)[1]) for k in z.files if k.startswith("layer_"))

    if args.layers:
        layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers = available_layers

    # Label maps.
    color_to_id: dict[str, int] = {}
    shape_to_id: dict[str, int] = {}
    conj_to_id: dict[str, int] = {}

    X_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    y_color: list[int] = []
    y_shape: list[int] = []
    y_conj: list[int] = []
    triplet_count_per_object: list[int] = []
    sample_id_per_object: list[int] = []

    for rec in tqdm(records, desc="Pool"):
        sample_id = int(rec["sample_id"])
        token_path = tokens_dir / f"sample_{sample_id:06d}.npz"
        if not token_path.exists():
            continue

        with np.load(token_path) as z:
            n_tokens = int(z["n_image_tokens"])
            h_patches = int(z["H_patches"])
            w_patches = int(z["W_patches"])
            h_patches, w_patches = _infer_grid(n_tokens, h_patches, w_patches)
            centers = _patch_centers(h_patches, w_patches, image_size=256)

            layer_embs = {layer: z[f"layer_{layer}"] for layer in layers}

        if any(layer_embs[layer].shape[0] != (h_patches * w_patches) for layer in layers):
            raise ValueError(
                f"Token count mismatch for sample {sample_id}: "
                f"expected {h_patches*w_patches} from grid, got {[layer_embs[l].shape[0] for l in layers]}"
            )

        for obj in rec["objects"]:
            color = obj["color"]
            shape = obj["shape"]

            if color not in color_to_id:
                color_to_id[color] = len(color_to_id)
            if shape not in shape_to_id:
                shape_to_id[shape] = len(shape_to_id)

            conj_key = f"{color}|||{shape}"
            if conj_key not in conj_to_id:
                conj_to_id[conj_key] = len(conj_to_id)

            mask_path = dataset_dir / obj["mask"]
            mask = np.asarray(Image.open(mask_path).convert("L"))
            xs = np.clip(centers[:, 0].round().astype(int), 0, 255)
            ys = np.clip(centers[:, 1].round().astype(int), 0, 255)
            assigned = mask[ys, xs] > 0

            if not assigned.any():
                x0, y0, x1, y1 = map(int, obj["box"])
                assigned = (centers[:, 0] >= x0) & (centers[:, 0] < x1) & (centers[:, 1] >= y0) & (centers[:, 1] < y1)

            if not assigned.any():
                raise ValueError(f"No patches assigned to object in sample {sample_id}; box={obj['box']}")

            for layer in layers:
                X_by_layer[layer].append(layer_embs[layer][assigned].mean(axis=0))

            y_color.append(color_to_id[color])
            y_shape.append(shape_to_id[shape])
            y_conj.append(conj_to_id[conj_key])
            triplet_count_per_object.append(int(rec["triplet_count"]))
            sample_id_per_object.append(sample_id)

    out_npz = out_dir / "embeddings.npz"
    arrays: dict[str, np.ndarray] = {
        "y_color": np.asarray(y_color, dtype=np.int64),
        "y_shape": np.asarray(y_shape, dtype=np.int64),
        "y_conj": np.asarray(y_conj, dtype=np.int64),
        "triplet_count_per_object": np.asarray(triplet_count_per_object, dtype=np.int64),
        "sample_id_per_object": np.asarray(sample_id_per_object, dtype=np.int64),
    }
    for layer in layers:
        arrays[f"X_{layer}"] = np.stack(X_by_layer[layer], axis=0).astype(np.float32)
    np.savez_compressed(out_npz, **arrays)

    label_maps = {
        "color_to_id": color_to_id,
        "shape_to_id": shape_to_id,
        "conj_to_id": conj_to_id,
    }
    (out_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

