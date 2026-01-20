import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score


def _layer_keys(npz: np.lib.npyio.NpzFile) -> list[int]:
    layers = []
    for k in npz.files:
        if k.startswith("X_"):
            layers.append(int(k.split("_", 1)[1]))
    return sorted(layers)


def _acc_by_group(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for g in np.unique(groups):
        mask = groups == g
        out[str(int(g))] = float(accuracy_score(y_true[mask], y_pred[mask])) if mask.any() else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probes on pooled object embeddings.")
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument(
        "--split_mode",
        type=str,
        default="sample_id",
        choices=["sample_id", "object"],
        help=(
            "How to form train/test splits. "
            "`sample_id` keeps all objects from the same image in the same split (recommended; default). "
            "`object` splits over objects directly, which can place objects from the same image in both train and test."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.embeddings_path) as z:
        layers = _layer_keys(z)
        y_color = z["y_color"]
        y_shape = z["y_shape"]
        y_conj = z["y_conj"]
        triplet = z["triplet_count_per_object"]
        sample_id = z.get("sample_id_per_object", None)
        if args.split_mode == "sample_id" and sample_id is None:
            raise KeyError(
                "Requested split_mode=sample_id but embeddings.npz does not contain `sample_id_per_object`. "
                "Re-run `probing/build_object_embeddings.py` to regenerate embeddings."
            )

        results: dict[str, dict] = {}
        for layer in layers:
            X = z[f"X_{layer}"]

            indices = np.arange(len(X))
            if args.split_mode == "sample_id":
                # Split by image (group), so all objects from the same rendered scene
                # go to either train or test (avoids leakage from near-duplicate context).
                splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_split, random_state=args.seed)
                idx_train, idx_test = next(splitter.split(indices, groups=sample_id))
            else:
                # Split by object: objects from the same image might land in both splits.
                idx_train, idx_test = train_test_split(
                    indices,
                    test_size=args.test_split,
                    random_state=args.seed,
                    stratify=y_conj if len(np.unique(y_conj)) > 1 else None,
                )

            layer_result: dict[str, dict] = {}
            for name, y in [("color", y_color), ("shape", y_shape), ("conjunction", y_conj)]:
                clf = LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    C=args.C,
                )
                clf.fit(X[idx_train], y[idx_train])
                y_pred = clf.predict(X[idx_test])
                layer_result[name] = {
                    "acc": float(accuracy_score(y[idx_test], y_pred)),
                    "acc_by_triplet_count": _acc_by_group(y[idx_test], y_pred, triplet[idx_test]),
                }

            results[str(layer)] = layer_result

    out_path = out_dir / "probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    # Print a small summary table.
    header = ["layer", "color_acc", "shape_acc", "conj_acc"]
    print("\t".join(header))
    for layer in layers:
        r = results[str(layer)]
        print(
            "\t".join(
                [
                    str(layer),
                    f"{r['color']['acc']:.3f}",
                    f"{r['shape']['acc']:.3f}",
                    f"{r['conjunction']['acc']:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
