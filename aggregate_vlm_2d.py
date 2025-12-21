#!/usr/bin/env python
"""
Aggregate 2D task results for a single model so they can be plotted later.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
# to extract the results from the model outputs.
# e.g The image contains five distinct shapes. <long description>. All shapes are separate and do not overlap. [5]
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_token(text: Optional[str]) -> Optional[str]:
    """Return the last token enclosed in square brackets from a response."""
    if not isinstance(text, str):
        return None
    matches = BRACKET_RE.findall(text)
    return matches[-1].strip() if matches else None


def token_to_bool(token: Optional[str]) -> Optional[bool]:
    """Convert a bracketed token into a boolean when possible."""
    if token is None:
        return None
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    return None


def token_to_int(token: Optional[str]) -> Optional[int]:
    """Convert a bracketed token into an integer when possible."""
    if token is None:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def normalize_label(value: Optional[str]) -> Optional[str]:
    """Clean simple labels (color/shape names) for downstream comparisons."""
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def parse_bool_literal(value: Any) -> Optional[bool]:
    """Parse loose boolean representations such as 'yes' or 0/1."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "t", "1", "yes"):
            return True
        if normalized in ("false", "f", "0", "no"):
            return False
    return None


def object_counter(objs: Iterable[Dict[str, str]]) -> Counter:
    """Count how many times each (color, shape) combination appears."""
    return Counter(
        (normalize_label(obj.get("color")), normalize_label(obj.get("shape")))
        for obj in objs
        if obj.get("color") and obj.get("shape")
    )


def count_object_errors(gt_objs: List[Dict[str, str]], pred_objs: List[Dict[str, str]]) -> int:
    """Compute symmetric difference size between ground-truth and predicted objects."""
    gt_counter = object_counter(gt_objs)
    pred_counter = object_counter(pred_objs)
    shared = sum(min(count, pred_counter[key]) for key, count in gt_counter.items())
    return len(gt_objs) + len(pred_objs) - 2 * shared


def _safe_literal_eval(text: str) -> Any:
    """Literal-eval helper that tolerates stray whitespace."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def parse_feature_sequence(value: Any) -> List[Dict[str, str]]:
    """Parse a list[dict(color, shape)] representation from CSV."""
    if isinstance(value, list):
        return value
    if isinstance(value, float) and math.isnan(value):
        return []
    if not isinstance(value, str):
        return []
    snippet = value.strip()
    if not snippet:
        return []
    if "[" in snippet and "]" in snippet:
        snippet = snippet[snippet.find("[") : snippet.rfind("]") + 1]
    # Try JSON first, then fall back to literal_eval.
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        data = _safe_literal_eval(snippet)
    if data is None:
        return []
    iterable: Iterable[Any]
    if isinstance(data, dict):
        iterable = data.values()
    else:
        iterable = data
    parsed: List[Dict[str, str]] = []
    for item in iterable:
        if not isinstance(item, dict):
            continue
        normalized = {str(k).strip().lower(): v for k, v in item.items()}
        color = normalize_label(normalized.get("color"))
        shape = normalize_label(normalized.get("shape"))
        if color and shape:
            parsed.append({"color": color, "shape": shape})
    return parsed


def parse_sequence_payload(text: Optional[str]) -> List[Dict[str, str]]:
    """Extract a list of {color, shape} dictionaries from a model response."""
    if not isinstance(text, str):
        return []
    snippet = text.strip()
    if "[" in snippet and "]" in snippet:
        snippet = snippet[snippet.find("[") : snippet.rfind("]") + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        data = _safe_literal_eval(snippet)
    if data is None:
        return []
    iterable = data.values() if isinstance(data, dict) else data
    parsed = []
    for item in iterable:
        if not isinstance(item, dict):
            continue
        normalized = {k.strip().lower(): v for k, v in item.items()}
        color = normalize_label(normalized.get("color"))
        shape = normalize_label(normalized.get("shape"))
        if color and shape:
            parsed.append({"color": color, "shape": shape})
    return parsed


def parse_rmts_dict(text: Optional[Any]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Parse nested RMTS dictionaries describing source/target pairs."""
    if isinstance(text, dict):
        data = text
    elif not isinstance(text, str):
        return {}
    else:
        snippet = text[text.find("{") : text.rfind("}") + 1] if "{" in text else text
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            data = _safe_literal_eval(snippet)
    if not isinstance(data, dict):
        return {}
    parsed: Dict[str, Dict[str, Dict[str, str]]] = {}
    for pair_name, objs in data.items():
        pair_key = str(pair_name).strip().lower()
        parsed[pair_key] = {}
        if isinstance(objs, dict):
            for obj_name, feat in objs.items():
                if isinstance(feat, dict):
                    parsed[pair_key][str(obj_name).strip().lower()] = {
                        "color": normalize_label(feat.get("color")),
                        "shape": normalize_label(feat.get("shape")),
                    }
    return parsed


def rmts_structures_match(gt: Dict[str, Dict[str, Dict[str, str]]],
                          pred: Dict[str, Dict[str, Dict[str, str]]]) -> bool:
    """Check whether predicted RMTS features exactly match the ground truth."""
    for pair in ("source", "target1", "target2"):
        gt_pair = gt.get(pair, {})
        pred_pair = pred.get(pair, {})
        if set(gt_pair.keys()) != set(pred_pair.keys()):
            return False
        for obj_name, gt_feat in gt_pair.items():
            pred_feat = pred_pair.get(obj_name)
            if pred_feat is None:
                return False
            if normalize_label(pred_feat.get("color")) != normalize_label(gt_feat.get("color")):
                return False
            if normalize_label(pred_feat.get("shape")) != normalize_label(gt_feat.get("shape")):
                return False
    return True


def count_feature_triplets(features: List[Dict[str, str]]) -> int:
    """Count ambiguous triplets (shared color & shape pairs) in a feature list."""
    total = 0
    for triple in combinations(features, 3):
        colors = [normalize_label(obj["color"]) for obj in triple]
        shapes = [normalize_label(obj["shape"]) for obj in triple]
        has_color_pair = len(set(colors)) < 3
        has_shape_pair = len(set(shapes)) < 3
        if has_color_pair and has_shape_pair:
            total += 1
    return total


def load_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Load a CSV if it exists, tagging the originating path for debugging."""
    # Helper keeps aggregation tolerant to missing task directories.
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["csv_path"] = str(path)
    return df


def summarize_visual_search(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    """Aggregate disjunctive/conjunctive results into accuracy by set size."""
    tasks = {
        "disjunctive_search": {"truth": "popout", "label": "disjunctive", "invert": False},
        # Control prompt asks “are all shapes same color?”, so True == no oddball.
        "disjunctive_search_control": {"truth": "popout", "label": "disjunctive_control", "invert": True},
        "conjunctive_search": {"truth": "incongruent", "label": "conjunctive", "invert": False},
    }
    frames = []
    for task, cfg in tasks.items():
        df = load_if_exists(results_root / task / f"{model}.csv")
        if df is None:
            continue
        df["prediction"] = df["response"].apply(lambda x: token_to_bool(extract_token(x)))
        df["n_objects"] = df["n_objects"].astype("Int64")
        truth_column = df[cfg["truth"]]
        if truth_column.dtype != bool:
            truth_column = truth_column.apply(parse_bool_literal)
        if cfg["invert"]:
            truth_column = truth_column.apply(lambda x: None if x is None else not x)
        df["correct_flag"] = df["prediction"] == truth_column
        grouped = (
            df.dropna(subset=["prediction"])
            .groupby("n_objects", as_index=False)["correct_flag"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "n_trials", "sum": "n_correct"})
        )
        grouped["accuracy"] = grouped["n_correct"] / grouped["n_trials"]
        grouped["task_name"] = task
        grouped["condition_label"] = cfg["label"]
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize_counting(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    """Compute accuracy per set size for the counting family of tasks."""
    tasks = {
        "counting_low_diversity": "low_entropy",
        "counting_high_diversity": "high_entropy",
        "counting_control": "medium_entropy_same_shape",
        "counting_control_shape": "medium_entropy_same_color",
        "counting_distinct": "control_distinct",
    }
    frames = []
    for task, label in tasks.items():
        df = load_if_exists(results_root / task / f"{model}.csv")
        if df is None:
            continue
        df["prediction"] = df["response"].apply(lambda x: token_to_int(extract_token(x)))
        df["n_objects"] = df["n_objects"].astype("Int64")
        df["correct_flag"] = df["prediction"] == df["n_objects"]
        grouped = (
            df.dropna(subset=["prediction"])
            .groupby("n_objects", as_index=False)["correct_flag"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "n_trials", "sum": "n_correct"})
        )
        grouped["accuracy"] = grouped["n_correct"] / grouped["n_trials"]
        grouped["task_name"] = task
        grouped["condition_label"] = label
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize_scene_description(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    """Measure scene-description errors grouped by number of objects/triplets."""
    task_dirs = ["scene_description", "scene_description_balanced"]
    frames = []
    for task_name in task_dirs:
        csv_path = results_root / task_name / f"{model}.csv"
        df = load_if_exists(csv_path)
        if df is None:
            continue
        df["features"] = df["features"].apply(parse_feature_sequence)
        if "n_objects" not in df.columns:
            df["n_objects"] = df["features"].apply(lambda feats: len(feats) if isinstance(feats, list) else np.nan)
        df["n_objects"] = df["n_objects"].astype("Int64")
        if "triplet_count" in df.columns:
            df["gt_triplets"] = df["triplet_count"].astype(int)
        else:
            df["gt_triplets"] = df["features"].apply(count_feature_triplets)
        df["prediction_objs"] = df["response"].apply(parse_sequence_payload)
        df["errors"] = df.apply(lambda row: count_object_errors(row["features"], row["prediction_objs"]), axis=1)
        grouped = (
            df.groupby(["n_objects", "gt_triplets"], as_index=False)["errors"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .rename(columns={"count": "n_trials", "mean": "mean_errors", "median": "median_errors"})
        )
        grouped["task_name"] = task_name
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize_rmts(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    """Summarize RMTS accuracies for each condition/subtask pairing."""
    subtasks = ("full", "relations", "features", "features2")
    frames = []
    for condition in ("unified", "decomposed"):
        for subtask in subtasks:
            csv_path = results_root / "rmts" / condition / subtask / f"{model}.csv"
            df = load_if_exists(csv_path)
            if df is None:
                continue
            if subtask == "full":
                df["prediction"] = df["response"].apply(lambda x: token_to_int(extract_token(x)))
                df["correct_flag"] = df["prediction"] == df["correct"].astype("Int64")
            elif subtask == "relations":
                df["prediction"] = df["response"].apply(lambda x: token_to_bool(extract_token(x)))
                truth = df["relation_value"].apply(parse_bool_literal)
                df["correct_flag"] = df["prediction"] == truth
            elif subtask == "features":
                df["prediction"] = df["response"].apply(lambda x: normalize_label(extract_token(x)))
                target = df["feature_value"].apply(normalize_label)
                df["correct_flag"] = df["prediction"] == target
            else:  # features2
                df["ground_truth"] = df["features"].apply(parse_rmts_dict)
                df["prediction_struct"] = df["response"].apply(parse_rmts_dict)
                df["correct_flag"] = df.apply(
                    lambda row: rmts_structures_match(
                        row["ground_truth"] if isinstance(row["ground_truth"], dict) else {},
                        row["prediction_struct"],
                    ),
                    axis=1,
                )
            grouped = df["correct_flag"].groupby(df.index).agg(["count", "sum"]).sum()
            n_trials = int(grouped["count"])
            n_correct = int(grouped["sum"])
            frames.append(
                pd.DataFrame(
                    {
                        "condition": [condition],
                        "subtask": [subtask],
                        "n_trials": [n_trials],
                        "n_correct": [n_correct],
                        "accuracy": [n_correct / n_trials if n_trials else np.nan],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    """Entry point aggregating every task family and writing CSV summaries."""
    parser = argparse.ArgumentParser(description="Aggregate 2D VLM task results.")
    parser.add_argument("--results-root", type=Path, default=Path("output/vlm/2D"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    visual = summarize_visual_search(args.results_root, args.model_name)
    if visual is not None:
        path = args.out_dir / f"{args.model_name}_visual_search.csv"
        visual.to_csv(path, index=False)
        print(f"Saved {path}")

    counting = summarize_counting(args.results_root, args.model_name)
    if counting is not None:
        path = args.out_dir / f"{args.model_name}_counting.csv"
        counting.to_csv(path, index=False)
        print(f"Saved {path}")

    scene = summarize_scene_description(args.results_root, args.model_name)
    if scene is not None:
        path = args.out_dir / f"{args.model_name}_scene_description.csv"
        scene.to_csv(path, index=False)
        print(f"Saved {path}")

    rmts = summarize_rmts(args.results_root, args.model_name)
    if rmts is not None:
        path = args.out_dir / f"{args.model_name}_rmts.csv"
        rmts.to_csv(path, index=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
