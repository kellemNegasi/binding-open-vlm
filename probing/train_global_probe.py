import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
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
    unique_vals, inverse = np.unique(group_values, return_inverse=True)
    # Sum predictions across labels per sample, then aggregate per group.
    sample_sum = (y_pred * mask).sum(axis=1, dtype=float)
    sample_count = mask.sum(axis=1, dtype=float)
    group_sum = np.bincount(inverse, weights=sample_sum, minlength=len(unique_vals))
    group_count = np.bincount(inverse, weights=sample_count, minlength=len(unique_vals))
    out: dict[str, float] = {}
    for idx, g in enumerate(unique_vals):
        if group_count[idx] == 0:
            out[str(int(g))] = float("nan")
        else:
            out[str(int(g))] = float(group_sum[idx] / group_count[idx])
    return out


def _mean_score_by_group(
    y_score: np.ndarray,
    mask: np.ndarray,
    group_values: np.ndarray,
) -> dict[str, float]:
    """Compute per-sample mean score by group, then unweighted average within group."""
    unique_vals, inverse = np.unique(group_values, return_inverse=True)
    sample_sum = (y_score * mask).sum(axis=1, dtype=float)
    sample_count = mask.sum(axis=1, dtype=float)
    per_sample_mean = np.divide(
        sample_sum,
        sample_count,
        out=np.full_like(sample_sum, np.nan, dtype=float),
        where=sample_count > 0,
    )
    valid = np.isfinite(per_sample_mean)
    group_sum = np.bincount(
        inverse[valid], weights=per_sample_mean[valid], minlength=len(unique_vals)
    )
    group_count = np.bincount(inverse[valid], minlength=len(unique_vals))
    out: dict[str, float] = {}
    for idx, g in enumerate(unique_vals):
        if group_count[idx] == 0:
            out[str(int(g))] = float("nan")
        else:
            out[str(int(g))] = float(group_sum[idx] / group_count[idx])
    return out


def _aggregate_group_metrics(
    per_seed: list[dict[str, float]],
) -> tuple[dict[str, float], dict[str, float]]:
    keys = sorted({k for d in per_seed for k in d.keys()}, key=lambda x: int(x))
    mean_out: dict[str, float] = {}
    std_out: dict[str, float] = {}
    for k in keys:
        vals = np.array([d.get(k, np.nan) for d in per_seed], dtype=float)
        finite = np.isfinite(vals)
        if not finite.any():
            mean_out[k] = float("nan")
            std_out[k] = float("nan")
            continue
        mean_out[k] = float(np.mean(vals[finite]))
        std_out[k] = float(np.std(vals[finite]))
    return mean_out, std_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-label probes on global scene embeddings.")
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated list of seeds to average over (overrides --seed).",
    )
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--ovr_n_jobs",
        type=int,
        default=1,
        help="Parallelize One-vs-Rest over labels. Use -1 for all cores.",
    )
    parser.add_argument(
        "--feature_jobs",
        type=int,
        default=1,
        help="Parallelize across features. Use -1 for all cores.",
    )
    parser.add_argument(
        "--progress_steps",
        action="store_true",
        help="Emit per-step progress within each seed (fit/predict/metrics/save).",
    )
    parser.add_argument(
        "--save_intermediates",
        action="store_true",
        default=True,
        help="Save per-seed, per-feature predictions and metadata for later aggregation.",
    )
    parser.add_argument(
        "--no_save_intermediates",
        dest="save_intermediates",
        action="store_false",
        help="Disable saving per-seed intermediates.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    with np.load(args.embeddings_path) as z:
        feature_keys = _feature_keys(z)
        y_present = z["y_present"]
        y_implied_absent = z["y_implied_absent"]
        y_non_implied_absent = z["y_non_implied_absent"]
        triplet = z["triplet_count_per_sample"]
        implied_counts = z["n_implied_absent_pairs_per_sample"]
        total_absent_counts = z.get("n_total_absent_pairs_per_sample", None)
        implied_density = z.get("implied_absent_density_per_sample", None)
        sample_id = z.get("sample_id", None)
        pair_colors = z.get("pair_colors", None)
        pair_shapes = z.get("pair_shapes", None)

        results: dict[str, dict] = {}
        intermediates_dir = out_dir / "intermediates"
        if args.save_intermediates:
            intermediates_dir.mkdir(parents=True, exist_ok=True)
        stratify_values = triplet if len(np.unique(triplet)) > 1 else None
        split_by_seed = {}
        for seed in seeds:
            split_by_seed[seed] = train_test_split(
                np.arange(len(triplet)),
                test_size=args.test_split,
                random_state=seed,
                stratify=stratify_values,
            )

        def process_feature(key_idx: int, key: str) -> tuple[str, dict]:
            X = z[key]
            print(f"[{key_idx}/{len(feature_keys)}] Feature {key}...", flush=True)

            per_seed_metrics = {
                "micro_f1": [],
                "macro_f1": [],
                "implied_fpr_overall": [],
                "non_implied_fpr_overall": [],
                "implied_fpr_by_triplet": [],
                "non_implied_fpr_by_triplet": [],
                "implied_fpr_by_implied_absent_count": [],
                "non_implied_fpr_by_implied_absent_count": [],
                "implied_mean_score_by_triplet": [],
                "non_implied_mean_score_by_triplet": [],
                "delta_mean_score_by_triplet": [],
                "implied_mean_score_by_implied_absent_count": [],
                "non_implied_mean_score_by_implied_absent_count": [],
                "delta_mean_score_by_implied_absent_count": [],
            }

            for seed_idx, seed in enumerate(seeds, start=1):
                print(f"  Seed {seed_idx}/{len(seeds)} (seed={seed})", flush=True)
                idx_train, idx_test = split_by_seed[seed]

                # Multi-label setting: one classifier per (color, shape) pair.
                clf = OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        C=args.C,
                        random_state=seed,
                    ),
                    n_jobs=args.ovr_n_jobs,
                )
                if args.progress_steps:
                    print(f"    [{key} seed={seed}] fit", flush=True)
                clf.fit(X[idx_train], y_present[idx_train])
                if args.progress_steps:
                    print(f"    [{key} seed={seed}] predict", flush=True)
                y_score = clf.predict_proba(X[idx_test])
                y_pred = (y_score >= args.threshold).astype(np.int8)

                if args.progress_steps:
                    print(f"    [{key} seed={seed}] metrics", flush=True)
                y_true = y_present[idx_test]
                implied_mask = y_implied_absent[idx_test].astype(bool)
                non_implied_mask = y_non_implied_absent[idx_test].astype(bool)

                per_seed_metrics["micro_f1"].append(
                    float(f1_score(y_true, y_pred, average="micro", zero_division=0))
                )
                per_seed_metrics["macro_f1"].append(
                    float(f1_score(y_true, y_pred, average="macro", zero_division=0))
                )
                per_seed_metrics["implied_fpr_overall"].append(
                    float(y_pred[implied_mask].mean()) if implied_mask.any() else float("nan")
                )
                per_seed_metrics["non_implied_fpr_overall"].append(
                    float(y_pred[non_implied_mask].mean()) if non_implied_mask.any() else float("nan")
                )
                per_seed_metrics["implied_fpr_by_triplet"].append(
                    _fpr_by_group(y_pred, implied_mask, triplet[idx_test])
                )
                per_seed_metrics["non_implied_fpr_by_triplet"].append(
                    _fpr_by_group(y_pred, non_implied_mask, triplet[idx_test])
                )
                per_seed_metrics["implied_fpr_by_implied_absent_count"].append(
                    _fpr_by_group(y_pred, implied_mask, implied_counts[idx_test])
                )
                per_seed_metrics["non_implied_fpr_by_implied_absent_count"].append(
                    _fpr_by_group(y_pred, non_implied_mask, implied_counts[idx_test])
                )

                implied_mean_by_triplet = _mean_score_by_group(
                    y_score, implied_mask, triplet[idx_test]
                )
                non_implied_mean_by_triplet = _mean_score_by_group(
                    y_score, non_implied_mask, triplet[idx_test]
                )
                per_seed_metrics["implied_mean_score_by_triplet"].append(implied_mean_by_triplet)
                per_seed_metrics["non_implied_mean_score_by_triplet"].append(
                    non_implied_mean_by_triplet
                )
                per_seed_metrics["delta_mean_score_by_triplet"].append(
                    {
                        k: float(implied_mean_by_triplet.get(k, np.nan))
                        - float(non_implied_mean_by_triplet.get(k, np.nan))
                        for k in implied_mean_by_triplet.keys()
                    }
                )

                implied_mean_by_count = _mean_score_by_group(
                    y_score, implied_mask, implied_counts[idx_test]
                )
                non_implied_mean_by_count = _mean_score_by_group(
                    y_score, non_implied_mask, implied_counts[idx_test]
                )
                per_seed_metrics["implied_mean_score_by_implied_absent_count"].append(
                    implied_mean_by_count
                )
                per_seed_metrics["non_implied_mean_score_by_implied_absent_count"].append(
                    non_implied_mean_by_count
                )
                per_seed_metrics["delta_mean_score_by_implied_absent_count"].append(
                    {
                        k: float(implied_mean_by_count.get(k, np.nan))
                        - float(non_implied_mean_by_count.get(k, np.nan))
                        for k in implied_mean_by_count.keys()
                    }
                )

                if args.save_intermediates:
                    if args.progress_steps:
                        print(f"    [{key} seed={seed}] save", flush=True)
                    out_payload = {
                        "idx_test": idx_test.astype(np.int32),
                        "triplet_count_per_sample": triplet[idx_test].astype(np.int32),
                        "n_implied_absent_pairs_per_sample": implied_counts[idx_test].astype(np.int32),
                        "y_present": y_present[idx_test].astype(np.int8),
                        "y_implied_absent": implied_mask.astype(np.int8),
                        "y_non_implied_absent": non_implied_mask.astype(np.int8),
                        "y_score": y_score.astype(np.float32),
                        "y_pred": y_pred.astype(np.int8),
                    }
                    if total_absent_counts is not None:
                        out_payload["n_total_absent_pairs_per_sample"] = total_absent_counts[
                            idx_test
                        ].astype(np.int32)
                    if implied_density is not None:
                        out_payload["implied_absent_density_per_sample"] = implied_density[
                            idx_test
                        ].astype(np.float32)
                    if sample_id is not None:
                        out_payload["sample_id"] = sample_id[idx_test].astype(np.int32)
                    if pair_colors is not None:
                        out_payload["pair_colors"] = pair_colors
                    if pair_shapes is not None:
                        out_payload["pair_shapes"] = pair_shapes

                    out_name = f"{key}_seed{seed}.npz"
                    np.savez_compressed(intermediates_dir / out_name, **out_payload)

            layer_result: dict[str, dict | float] = {
                "micro_f1": float(np.nanmean(per_seed_metrics["micro_f1"])),
                "macro_f1": float(np.nanmean(per_seed_metrics["macro_f1"])),
                "implied_fpr_overall": float(np.nanmean(per_seed_metrics["implied_fpr_overall"])),
                "non_implied_fpr_overall": float(np.nanmean(per_seed_metrics["non_implied_fpr_overall"])),
            }

            for name in [
                "implied_fpr_by_triplet",
                "non_implied_fpr_by_triplet",
                "implied_fpr_by_implied_absent_count",
                "non_implied_fpr_by_implied_absent_count",
                "implied_mean_score_by_triplet",
                "non_implied_mean_score_by_triplet",
                "delta_mean_score_by_triplet",
                "implied_mean_score_by_implied_absent_count",
                "non_implied_mean_score_by_implied_absent_count",
                "delta_mean_score_by_implied_absent_count",
            ]:
                if name not in per_seed_metrics:
                    continue
                mean_out, std_out = _aggregate_group_metrics(per_seed_metrics[name])
                layer_result[name] = mean_out
                layer_result[f"{name}_std"] = std_out

            if len(seeds) > 1:
                layer_result["micro_f1_std"] = float(np.nanstd(per_seed_metrics["micro_f1"]))
                layer_result["macro_f1_std"] = float(np.nanstd(per_seed_metrics["macro_f1"]))
                layer_result["implied_fpr_overall_std"] = float(
                    np.nanstd(per_seed_metrics["implied_fpr_overall"])
                )
                layer_result["non_implied_fpr_overall_std"] = float(
                    np.nanstd(per_seed_metrics["non_implied_fpr_overall"])
                )

            return str(key), layer_result

        feature_results = Parallel(n_jobs=args.feature_jobs, prefer="threads")(
            delayed(process_feature)(idx, key) for idx, key in enumerate(feature_keys, start=1)
        )

        for key, layer_result in feature_results:
            results[key] = layer_result

    results["_meta"] = {
        "seeds": seeds,
        "threshold": args.threshold,
        "C": args.C,
        "save_intermediates": args.save_intermediates,
    }

    out_path = out_dir / "probe_results.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    per_sample_path = out_dir / "probe_results_per_sample.json"
    per_sample_path.write_text(json.dumps(results, indent=2, sort_keys=True))

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
