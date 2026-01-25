# probing/extract_image_tokens.py
import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from probing.model_introspect import InternVL2Introspector


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--model_tag", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    out_dir = Path(args.out_dir) / "tokens"
    out_dir.mkdir(parents=True, exist_ok=True)

    introspector = InternVL2Introspector(args.model_key)

    # EXPECTED DATASET STRUCTURE:
    # dataset_dir/images/*.jpg (or png)
    image_dir = Path(args.dataset_dir) / "images"
    images = sorted(image_dir.glob("*"))

    if args.max_samples is not None:
        images = images[: args.max_samples]

    assert len(images) > 0, f"No images found in {image_dir}"

    for idx, img_path in enumerate(tqdm(images, desc="Extract")):
        image = Image.open(img_path)

        hs = introspector.extract(
            image=image,
            prompt="",   # not used for vision-only
            layers=layers,
        )

        out_path = out_dir / f"sample_{idx:06d}.npz"
        np.savez(
            out_path,
            **{f"layer_{k}": v.numpy() for k, v in hs.per_layer.items()},
            n_image_tokens=hs.n_image_tokens,
            h_patches=hs.h_patches,
            w_patches=hs.w_patches,
        )


if __name__ == "__main__":
    main()
