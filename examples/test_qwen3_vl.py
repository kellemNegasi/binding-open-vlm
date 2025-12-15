#!/usr/bin/env python3
"""Minimal lmdeploy test for Qwen3-VL-30B-A3B-Instruct with auto-download."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image
from lmdeploy import pipeline

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[1] / "model-weights" / "Qwen3-VL-30B-A3B-Instruct"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face repo id for the model (default: %(default)s).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional specific revision/commit of the Hugging Face model.",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_MODEL_DIR),
        help="Path to the weights directory (default: %(default)s).",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to an image file the model should describe."
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail.",
        help="Text prompt to pair with the image."
    )
    parser.add_argument(
        "--output",
        default="qwen3_test_output.txt",
        help="File to store the generated response."
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    weights_path = ensure_weights(Path(args.weights), args.model_id, args.revision)

    print(f"Loading lmdeploy pipeline from: {weights_path}")
    pipe = pipeline(model_path=str(weights_path))

    print("Running inference...")
    result = pipe([(args.prompt, Image.open(image_path))])

    print("Saving output...")
    response = result[0]
    response_text = response.text if hasattr(response, "text") else str(response)
    Path(args.output).write_text(response_text)
    print(f"Response written to {args.output}")
    print("Generated text:\n", response_text)



if __name__ == "__main__":
    main()
