#!/usr/bin/env python
"""
Aggregate 2D task results for a single model so they can be plotted later.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


def extract_token(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str):
        return None
    matches = BRACKET_RE.findall(text)
    return matches[-1].strip() if matches else None


def token_to_bool(token: Optional[str]) -> Optional[bool]:
    if token is None:
        return None
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    return None


def token_to_int(token: Optional[str]) -> Optional[int]:
    if token is None:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def normalize_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def object_counter(objs: Iterable[Dict[str, str]]) -> Counter:
    return Counter(
        (normalize_label(obj.get("color")), normalize_label(obj.get("shape")))
        for obj in objs
        if obj.get("color") and obj.get("shape")
    )


def count_object_errors(gt_objs: List[Dict[str, str]], pred_objs: List[Dict[str, str]]) -> int:
    gt_counter = object_counter(gt_objs)
    pred_counter = object_counter(pred_objs)
    shared = sum(min(count, pred_counter[key]) for key, count in gt_counter.items())
    return len(gt_objs) + len(pred_objs) - 2 * shared


def parse_sequence_payload(text: Optional[str]) -> List[Dict[str, str]]:
    if not isinstance(text, str):
        return []
    snippet = text.strip()
    if "[" in snippet and "]" in snippet:
        snippet = snippet[snippet.find("[") : snippet.rfind("]") + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(snippet)
        except (ValueError, SyntaxError):
            return []
    if isinstance(data, dict):
        iterable = data.values()
    else:
        iterable = data
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


def parse_rmts_dict(text: Optional[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    if not isinstance(text, str):
        return {}
    snippet = text[text.find("{") : text.rfind("}") + 1] if "{" in text else text
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(snippet)
        except (ValueError, SyntaxError):
            return {}
    if not isinstance(data, dict):
        return {}
    parsed: Dict[str, Dict[str, Dict[str, str]]] = {}
    for pair_name, objs in data.items():
        pair_key = pair_name.strip().lower()
        parsed[pair_key] = {}
        if isinstance(objs, dict):
            for obj_name, feat in objs.items():
                if isinstance(feat, dict):
                    parsed[pair_key][obj_name.strip().lower()] = {
                        "color": normalize_label(feat.get("color")),
                        "shape": normalize_label(feat.get("shape")),
                    }
    return parsed


def rmts_structures_match(gt: Dict[str, Dict[str, Dict[str, str]]],
                          pred: Dict[str, Dict[str, Dict[str, str]]]) -> bool:
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
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["csv_path"] = str(path)
    return df


def summarize_visual_search(results_root: Path, model: str) -> Optional[pd.DataFrame]:
    tasks = {
        "disjunctive_search": {"truth": "popout", "label": "disjunctive"},
        "disjunctive_search_control": {"truth": "popout", "label": "disjunctive_control"},
        "conjunctive_search": {"truth": "incongruent", "label": "conjunctive"},
    }
    frames = []
    for task, cfg in tasks.items():
        df = load_if_exists(results_root / task / f"{model}.csv")
        if df is None:
            continue
        df["prediction"] = df["response"].apply(lambda x: token_to_bool(extract_token(x)))
        df["n_objects"] = df["n_objects"].astype("Int64")
        df["correct_flag"] = df["prediction"] == df[cfg["truth"]]
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
    csv_path = results_root / "scene_description" / f"{model}.csv"
    df = load_if_exists(csv_path)
    if df is None:
        return None
    df["features"] = df["features"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["gt_triplets"] = df["features"].apply(count_feature_triplets)
    df["prediction_objs"] = df["response"].apply(parse_sequence_payload)
    df["errors"] = df.apply(lambda row: count_object_errors(row["features"], row["prediction_objs"]), axis=1)
    grouped = (
        df.groupby(["n_objects", "gt_triplets"], as_index=False)["errors"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "n_trials", "mean": "mean_errors", "median": "median_errors"})
    )
    grouped["task_name"] = "scene_description"
    return grouped


def summarize_rmts(results_root: Path, model: str) -> Optional[pd.DataFrame]:
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
                truth = df["relation_value"].astype(bool)
                df["correct_flag"] = df["prediction"] == truth
            elif subtask == "features":
                df["prediction"] = df["response"].apply(lambda x: normalize_label(extract_token(x)))
                target = df["feature_value"].apply(normalize_label)
                df["correct_flag"] = df["prediction"] == target
            else:  # features2
                df["ground_truth"] = df["features"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                df["prediction_struct"] = df["response"].apply(parse_rmts_dict)
                df["correct_flag"] = df.apply(
                    lambda row: rmts_structures_match(
                        {k: {kk:kdict for kk,kdict in v.items()} for k,v in row["ground_truth"].items()},
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
