import argparse
import json
from pathlib import Path

from PIL import Image

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from probing.model_introspect import _infer_image_token_indices, _maybe_prepend_image_token, load_model_from_hydra


def _load_first_dataset_image(dataset_dir: Path) -> Image.Image | None:
    meta_path = dataset_dir / "meta.jsonl"
    if not meta_path.exists():
        return None
    with meta_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            image_path = dataset_dir / rec["image"]
            if image_path.exists():
                return Image.open(image_path).convert("RGB")
    return None


def _contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
            continue
        ranges.append((start, prev))
        start = prev = i
    ranges.append((start, prev))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Debug helper: run the HF processor for a VLM, then report where image tokens and text tokens land in "
            "the combined sequence. This helps verify that probing extracts hidden states from the intended image tokens."
        )
    )
    parser.add_argument("--model", type=str, required=True, help="Hydra model key, e.g. qwen3_vl_30b")
    parser.add_argument("--prompt_path", type=str, default="prompts/scene_description_2D_parse.txt")
    parser.add_argument("--image_path", type=str, default=None, help="Optional path to an image.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Optional dataset dir containing meta.jsonl; if provided and image_path is omitted, uses first image.",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "xpu"])
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument(
        "--print_window",
        type=int,
        default=20,
        help="How many tokens to print around the vision start/end markers (0 disables).",
    )
    args = parser.parse_args()

    prompt_path = Path(args.prompt_path)
    if not prompt_path.exists():
        prompt_path = root / args.prompt_path
    prompt = prompt_path.read_text()
    prompt = _maybe_prepend_image_token(prompt)

    image: Image.Image | None = None
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")
    elif args.dataset_dir:
        image = _load_first_dataset_image(Path(args.dataset_dir))
    if image is None:
        image = Image.new("RGB", (256, 256), color=(255, 255, 255))

    model = load_model_from_hydra(args.model)
    # Only need the processor/tokenizer to inspect input_ids, so avoid loading HF model weights.
    if not getattr(model, "weights_path", None):
        raise ValueError(f"Model wrapper {type(model).__name__} does not expose `weights_path`; cannot load processor.")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model.weights_path, trust_remote_code=True)

    tokenizer = processor.tokenizer

    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    seq_len = int(input_ids.shape[1])

    image_token_indices = _infer_image_token_indices(tokenizer, input_ids)
    image_idx_list = [int(i) for i in image_token_indices.detach().cpu().tolist()]
    image_ranges = _contiguous_ranges(image_idx_list)

    id_to_tok = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    image_toks = [id_to_tok[i] for i in image_idx_list]
    unique_image_toks = sorted(set(image_toks))

    print(f"model={args.model}")
    print(f"seq_len={seq_len}")
    print(f"n_image_tokens={len(image_idx_list)}")
    print(f"image_index_ranges={image_ranges}")
    print(f"unique_image_token_strings={unique_image_toks}")

    # Show where a few known multimodal boundary tokens land, if they exist.
    boundary_tokens = ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<image>"]
    for tok in boundary_tokens:
        try:
            tid = tokenizer.encode(tok, add_special_tokens=False)
        except Exception:
            continue
        if not (isinstance(tid, list) and len(tid) == 1):
            continue
        tid = int(tid[0])
        positions = [i for i, x in enumerate(input_ids[0].detach().cpu().tolist()) if int(x) == tid]
        if positions:
            print(f"{tok} id={tid} positions={positions[:20]}{'...' if len(positions) > 20 else ''}")

    if args.print_window > 0:
        # Print token snippets around the first and last image-token ranges (most common layout).
        for label, pos in [("first_image_start", image_ranges[0][0]), ("last_image_end", image_ranges[-1][1])]:
            lo = max(0, pos - args.print_window)
            hi = min(seq_len, pos + args.print_window + 1)
            snippet = [(i, id_to_tok[i]) for i in range(lo, hi)]
            print(f"\n[{label}] window={lo}:{hi}")
            print(" ".join([f"{i}:{t}" for i, t in snippet]))

    # Quick sanity check: text token indices are everything except the image tokens.
    image_idx_set = set(image_idx_list)
    n_text_like = seq_len - len(image_idx_set)
    print(f"n_non_image_tokens={n_text_like} (includes special + text)")


if __name__ == "__main__":
    main()
