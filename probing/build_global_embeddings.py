import argparse
import json
from pathlib import Path

import numpy as np


def _layer_keys(npz: np.lib.npyio.NpzFile) -> list[int]:
    layers = []
    for k in npz.files:
        if k.startswith("layer_"):
            layers.append(int(k.split("_", 1)[1]))
    return sorted(layers)


def _pool_tokens(tokens: np.ndarray, mode: str) -> np.ndarray:
    # If extraction already saved pooled embeddings (1D), keep as-is.
    if tokens.ndim == 1:
        return tokens
    if mode == "mean":
        return tokens.mean(axis=0)
    if mode == "first":
        return tokens[0]
    if mode == "last":
        return tokens[-1]
    raise ValueError(f"Unsupported pool mode: {mode}")


def _implied_absent_pairs(objects: list[dict], colors: list[str], shapes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (present, implied-absent) labels for every (color, shape) pair.

    Implied-absent pairs are absent but supported by:
      1) the color appearing with another shape, and
      2) the shape appearing with another color.
    """
    present = {(obj["color"], obj["shape"]) for obj in objects}
    colors_present = {obj["color"] for obj in objects}
    shapes_present = {obj["shape"] for obj in objects}

    y_present = []
    y_implied = []
    for color in colors:
        for shape in shapes:
            pair = (color, shape)
            if pair in present:
                y_present.append(1)
                y_implied.append(0)
                continue

            if color not in colors_present or shape not in shapes_present:
                y_present.append(0)
                y_implied.append(0)
                continue

            # Swap condition: evidence for (color, *) and (*, shape) without seeing (color, shape).
            has_color_other_shape = any(
                obj["color"] == color and obj["shape"] != shape for obj in objects
            )
            has_shape_other_color = any(
                obj["shape"] == shape and obj["color"] != color for obj in objects
            )
            implied = has_color_other_shape and has_shape_other_color
            y_present.append(0)
            y_implied.append(1 if implied else 0)

    return np.asarray(y_present, dtype=np.int8), np.asarray(y_implied, dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build global scene embeddings and labels for illusory conjunction probing."
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--tokens_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated list of layers to include.")
    parser.add_argument(
        "--pool",
        type=str,
        default="mean",
        choices=["mean", "first", "last"],
        help="Pooling strategy over image tokens.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    tokens_dir = Path(args.tokens_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = dataset_dir / "meta.jsonl"
    records = []
    with meta_path.open() as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        raise ValueError(f"No records found in {meta_path}")

    # Discover global color/shape vocab and layer keys from the first token file.
    colors = sorted({obj["color"] for rec in records for obj in rec["objects"]})
    shapes = sorted({obj["shape"] for rec in records for obj in rec["objects"]})

    first_id = int(records[0]["sample_id"])
    first_path = tokens_dir / f"sample_{first_id:06d}.npz"
    if not first_path.exists():
        raise FileNotFoundError(f"Missing token file: {first_path}")

    with np.load(first_path) as z:
        all_layers = _layer_keys(z)
        cls_layers = []
        for k in z.files:
            if k.startswith("cls_"):
                cls_layers.append(int(k.split("_", 1)[1]))
        cls_layers = sorted(set(cls_layers))

    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = all_layers

    X_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    X_by_cls: dict[int, list[np.ndarray]] = {layer: [] for layer in cls_layers}
    y_present_list = []
    y_implied_list = []
    triplet_counts = []
    sample_ids = []

    for rec in records:
        sample_id = int(rec["sample_id"])
        token_path = tokens_dir / f"sample_{sample_id:06d}.npz"
        if not token_path.exists():
            raise FileNotFoundError(f"Missing token file: {token_path}")

        with np.load(token_path) as z:
            for layer in layers:
                tokens = z[f"layer_{layer}"]
                X_by_layer[layer].append(_pool_tokens(tokens, args.pool))
            for layer in cls_layers:
                cls_key = f"cls_{layer}"
                if cls_key in z:
                    X_by_cls[layer].append(np.asarray(z[cls_key], dtype=np.float32))

        y_present, y_implied = _implied_absent_pairs(rec["objects"], colors, shapes)
        y_present_list.append(y_present)
        y_implied_list.append(y_implied)
        triplet_counts.append(int(rec.get("triplet_count", 0)))
        sample_ids.append(sample_id)

    # Label matrices are [n_samples, n_pairs], where n_pairs = n_colors * n_shapes.
    y_present = np.stack(y_present_list, axis=0)
    y_implied_absent = np.stack(y_implied_list, axis=0)
    # Absent and not implied is the remaining complement.
    y_non_implied_absent = (1 - y_present) * (1 - y_implied_absent)
    n_implied_absent_pairs = y_implied_absent.sum(axis=1).astype(np.int32)
    n_total_absent_pairs = (1 - y_present).sum(axis=1).astype(np.int32)
    implied_absent_density = np.divide(
        n_implied_absent_pairs,
        np.maximum(n_total_absent_pairs, 1),
        dtype=np.float32,
    )

    out_npz = out_dir / "global_embeddings.npz"
    payload = {
        "y_present": y_present.astype(np.int8),
        "y_implied_absent": y_implied_absent.astype(np.int8),
        "y_non_implied_absent": y_non_implied_absent.astype(np.int8),
        "triplet_count_per_sample": np.asarray(triplet_counts, dtype=np.int32),
        "n_implied_absent_pairs_per_sample": n_implied_absent_pairs,
        "n_total_absent_pairs_per_sample": n_total_absent_pairs,
        "implied_absent_density_per_sample": implied_absent_density,
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        # Pair index i corresponds to (pair_colors[i], pair_shapes[i]).
        "pair_colors": np.asarray([c for c in colors for _ in shapes], dtype="U"),
        "pair_shapes": np.asarray([s for _ in colors for s in shapes], dtype="U"),
    }
    for layer, xs in X_by_layer.items():
        payload[f"X_{layer}"] = np.stack(xs, axis=0)
    for layer, xs in X_by_cls.items():
        if xs:
            payload[f"X_cls_{layer}"] = np.stack(xs, axis=0)

    np.savez_compressed(out_npz, **payload)
    print(f"Wrote {out_npz}")


if __name__ == "__main__":
    main()
