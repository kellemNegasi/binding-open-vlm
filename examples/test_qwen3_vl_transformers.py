#!/usr/bin/env python3
"""Run Qwen3-VL-30B-A3B-Instruct via Hugging Face Transformers (no lmdeploy)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "models" / "Qwen3-VL-30B-A3B-Instruct"
)


def ensure_weights(weights_path: Path, model_id: str, revision: str | None) -> Path:
    """Download model weights if they do not already exist on disk."""
    weights_path = weights_path.expanduser().resolve()
    if weights_path.exists() and any(weights_path.iterdir()):
        print(f"Using locally cached weights at: {weights_path}")
        return weights_path

    print(f"Downloading {model_id} to {weights_path} ...")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=str(weights_path),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return weights_path


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move tensor inputs onto the specified device."""
    out = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face repo id for the model (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional specific revision/commit for the model.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the weights directory (default: %(default)s).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to describe.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Text prompt that accompanies the image.",
    )
    parser.add_argument(
        "--output",
        default="qwen3_transformers_output.txt",
        help="Destination file for the generated response.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to sample.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype used when loading the model.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="device_map argument forwarded to `from_pretrained` (default: auto).",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help="Optional per-GPU memory cap, e.g. 28GiB (applies to every visible GPU).",
    )
    parser.add_argument(
        "--cpu-memory",
        default=None,
        help="Optional CPU memory budget for offload, e.g. 256GiB.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    weights_path = ensure_weights(Path(args.weights), args.model_id, args.revision)

    dtype = parse_dtype(args.dtype)
    print(f"Loading Transformers model from: {weights_path}")
    max_memory = None
    if args.max_gpu_memory or args.cpu_memory:
        max_memory = {}
        if args.max_gpu_memory:
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError(
                    "--max-gpu-memory was provided but CUDA is unavailable."
                )
            for idx in range(num_gpus):
                max_memory[f"cuda:{idx}"] = args.max_gpu_memory
        if args.cpu_memory:
            max_memory["cpu"] = args.cpu_memory
    model = AutoModelForVision2Seq.from_pretrained(
        weights_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        weights_path,
        trust_remote_code=True,
    )

    image = Image.open(image_path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    )
    if args.device_map != "cpu":
        target_device = model.device
    else:
        target_device = torch.device("cpu")
    inputs = move_to_device(inputs, target_device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
        )
    text_output = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    Path(args.output).write_text(text_output)
    print(f"Response written to {args.output}")
    print("Generated text:\n", text_output)


if __name__ == "__main__":
    main()
