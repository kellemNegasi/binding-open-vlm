#!/usr/bin/env python3
# run as python export_first_rows.py output/vlm/2D sample_answers.json
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

_SANITIZE = re.compile(r"[^0-9a-zA-Z]+")


def coerce_value(value: Optional[str]):
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            continue
    return text


def read_first_row(csv_path: Path) -> Dict[str, object]:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        try:
            row = next(reader)
        except StopIteration:
            raise ValueError(f"{csv_path} has no data rows.") from None
        return {column: coerce_value(value) for column, value in row.items()}


def derive_task_key(csv_path: Path, root: Path) -> str:
    rel = csv_path.resolve().relative_to(root.resolve())
    dirs = rel.parts[:-1]
    if dirs:
        parts = dirs
    else:
        parts = (Path(rel.name).stem,)
    normalized = [
        _SANITIZE.sub("_", part).strip("_").lower()
        for part in parts
        if part.strip()
    ]
    if not normalized:
        normalized = [_SANITIZE.sub("_", csv_path.stem).strip("_").lower()]
    return "_".join(filter(None, normalized))


def gather_csvs(root: Path):
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the first row of every CSV under a root directory."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory containing CSV outputs (e.g., output/vlm/2D).",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Destination JSON file (e.g., sample_answers.json).",
    )
    args = parser.parse_args()

    root = args.root_dir
    if not root.is_dir():
        sys.exit(f"error: {root} is not a directory.")
    csv_files = gather_csvs(root)
    if not csv_files:
        sys.exit(f"error: no *.csv files found under {root}")

    summary = {}
    for csv_file in csv_files:
        key = derive_task_key(csv_file, root)
        summary[key] = read_first_row(csv_file)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as fh:
        json.dump(summary, fh, indent=4)
    print(f"Wrote {len(summary)} entr{'y' if len(summary) == 1 else 'ies'} to {args.output_json}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"error: {exc}")
