import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def _feature_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    keys = [k for k in npz.files if k.startswith("X_")]
    return sorted(keys)


def _fpr_by_group(
    y_pred: np.ndarray,
    mask: np.ndarray,
    group_values: np.ndarray,
) -> dict[str, float]:
    """Compute false-positive rate by grouping value.

    `mask` selects which labels are considered absent (implied or non-implied).
    """
    out: dict[str, float] = {}
    for g in np.unique(group_values):
        g_mask = group_values == g
        select = g_mask[:, None] & mask
        # Since select refers to absent pairs only, mean(pred) is the false-positive rate.
        out[str(int(g))] = float(y_pred[select].mean()) if select.any() else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-label probes on global scene embeddings.")
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.embeddings_path) as z:
        feature_keys = _feature_keys(z)
        y_present = z["y_present"]
        y_implied_absent = z["y_implied_absent"]
        y_non_implied_absent = z["y_non_implied_absent"]
        triplet = z["triplet_count_per_sample"]
        implied_counts = z["n_implied_absent_pairs_per_sample"]

        results: dict[str, dict] = {}
        for key in feature_keys:
            X = z[key]

            idx_train, idx_test = train_test_split(
                np.arange(len(X)),
                test_size=args.test_split,
                random_state=args.seed,
                stratify=triplet if len(np.unique(triplet)) > 1 else None,
            )

            # Multi-label setting: one classifier per (color, shape) pair.
            clf = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    C=args.C,
                )
            )
            clf.fit(X[idx_train], y_present[idx_train])
            y_score = clf.predict_proba(X[idx_test])
            y_pred = (y_score >= args.threshold).astype(np.int8)

            y_true = y_present[idx_test]
            implied_mask = y_implied_absent[idx_test].astype(bool)
            non_implied_mask = y_non_implied_absent[idx_test].astype(bool)

            layer_result = {
                "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "implied_fpr_overall": float(y_pred[implied_mask].mean()) if implied_mask.any() else float("nan"),
                "non_implied_fpr_overall": float(y_pred[non_implied_mask].mean())
                if non_implied_mask.any()
                else float("nan"),
                "implied_fpr_by_triplet": _fpr_by_group(
                    y_pred, implied_mask, triplet[idx_test]
                ),
                "non_implied_fpr_by_triplet": _fpr_by_group(
                    y_pred, non_implied_mask, triplet[idx_test]
                ),
                "implied_fpr_by_implied_absent_count": _fpr_by_group(
                    y_pred, implied_mask, implied_counts[idx_test]
                ),
                "non_implied_fpr_by_implied_absent_count": _fpr_by_group(
                    y_pred, non_implied_mask, implied_counts[idx_test]
                ),
            }

            results[str(key)] = layer_result

    out_path = out_dir / "probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))

    header = ["feature", "micro_f1", "macro_f1", "implied_fpr", "non_implied_fpr"]
    print("\t".join(header))
    for key in feature_keys:
        r = results[str(key)]
        print(
            "\t".join(
                [
                    str(key),
                    f"{r['micro_f1']:.3f}",
                    f"{r['macro_f1']:.3f}",
                    f"{r['implied_fpr_overall']:.3f}",
                    f"{r['non_implied_fpr_overall']:.3f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
