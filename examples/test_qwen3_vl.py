#!/usr/bin/env python3
"""Minimal lmdeploy test for Qwen3-VL-30B-A3B-Instruct."""
import argparse
from pathlib import Path
from PIL import Image
from lmdeploy import pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the Qwen3-VL-30B-A3B-Instruct weights directory "
             "(e.g., /models/Qwen3-VL-30B-A3B-Instruct)."
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

    print(f"Loading lmdeploy pipeline from: {args.weights}")
    pipe = pipeline(model_path=args.weights)

    print("Running inference...")
    result = pipe([(args.prompt, Image.open(image_path))])

    print("Saving output...")
    Path(args.output).write_text(result[0])
    print(f"Response written to {args.output}")
    print("Generated text:\n", result[0])


if __name__ == "__main__":
    main()
