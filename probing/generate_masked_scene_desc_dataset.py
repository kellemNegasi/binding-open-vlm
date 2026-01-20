import argparse
import json
import random
from pathlib import Path

import numpy as np
from matplotlib import colors as mcolors
from PIL import Image
from tqdm import tqdm
import yaml

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from tasks.scene_description import SceneDescriptionBalanced
from utils import color_shape, place_shapes, resize


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _mask_to_box(mask_img: Image.Image) -> list[int]:
    mask = np.asarray(mask_img)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [x0, y0, x1, y1]


def _load_task_defaults(config_task_yaml: str | None) -> dict:
    if config_task_yaml is None:
        config_task_yaml = str(root / "config" / "task" / "scene_description_BALANCED.yaml")
    cfg = yaml.safe_load(Path(config_task_yaml).read_text())
    return {
        "color_names": cfg["color_names"],
        "shape_names": cfg["shape_names"],
        "shape_inds": cfg["shape_inds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a masked SceneDescriptionBalanced 2D probing dataset.")
    parser.add_argument("--out_dir", type=str, default="data/probing/scene_description_balanced_2d")
    parser.add_argument("--n_objects", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_task_yaml", type=str, default=None)
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip generation if meta.jsonl already exists with at least n_trials rows.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)

    defaults = _load_task_defaults(args.config_task_yaml)
    color_names = np.array(defaults["color_names"])
    shape_names = np.array(defaults["shape_names"])
    shape_inds = np.array(defaults["shape_inds"])

    shapes = np.load(str(root / "imgs.npy"))
    shapes = shapes[shape_inds]

    # Use the original SceneDescriptionBalanced sampling logic without writing into benchmark folders.
    sampler = SceneDescriptionBalanced.__new__(SceneDescriptionBalanced)
    trial_df = SceneDescriptionBalanced.generate_trial_df(sampler, color_names, shape_names, args.n_objects, args.n_trials)
    trial_df["shape_inds"] = trial_df.shape_vecs.apply(lambda x: np.where(x)[1])

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.jsonl"
    if args.skip_if_exists and meta_path.exists():
        existing_rows = sum(1 for _ in meta_path.open("r"))
        if existing_rows >= args.n_trials:
            return

    with meta_path.open("w") as f:
        for sample_id, trial in tqdm(trial_df.iterrows(), total=len(trial_df)):
            colors = np.array(trial.colors)
            shape_inds_local = np.array(trial.shape_inds, dtype=int)
            shape_imgs = shapes[shape_inds_local]

            rgb_codes = np.array([mcolors.to_rgba(color)[:-1] for color in colors])
            colored_imgs = [color_shape(img.astype(np.float32), rgb) for img, rgb in zip(shape_imgs, rgb_codes)]
            resized_imgs = [resize(img, img_size=args.img_size) for img in colored_imgs]

            canvas_img, masks, _boxes = place_shapes(resized_imgs, img_size=args.img_size + 10, return_masks=True)

            image_rel = Path("images") / f"sample_{sample_id:06d}.png"
            canvas_img.save(images_dir / image_rel.name)

            objects_meta = []
            for j, (mask_img, color, shape_ind) in enumerate(zip(masks, colors, shape_inds_local)):
                mask_rel = Path("masks") / f"sample_{sample_id:06d}_obj{j:02d}.png"
                mask_img.save(masks_dir / mask_rel.name)
                box = _mask_to_box(mask_img)
                objects_meta.append(
                    {
                        "color": str(color),
                        "shape": str(shape_names[shape_ind]),
                        "mask": str(mask_rel.as_posix()),
                        "box": box,
                    }
                )

            record = {
                "sample_id": int(sample_id),
                "image": str(image_rel.as_posix()),
                "triplet_count": int(trial.triplet_count),
                "n_objects": int(args.n_objects),
                "n_shapes": int(trial.n_shapes),
                "n_colors": int(trial.n_colors),
                "objects": objects_meta,
            }
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
