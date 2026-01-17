import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from probing.model_introspect import extract_image_hidden_states, load_model_from_hydra


def _parse_layers(s: str) -> list[int]:
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _load_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-layer image-token hidden states for a probing dataset.")
    parser.add_argument("--dataset_dir", type=str, default="data/probing/scene_description_balanced_2d")
    parser.add_argument("--out_dir", type=str, default="data/probing/scene_description_balanced_2d_out")
    parser.add_argument("--model", type=str, required=True, help="Hydra model key, e.g. qwen3_vl_30b")
    parser.add_argument("--model_config", type=str, default=None, help="Optional config/model/*.yaml path (used to infer model key)")
    parser.add_argument("--prompt_path", type=str, default="prompts/scene_description_2D_parse.txt")
    parser.add_argument("--prompt_variant", type=str, default="auto", choices=["auto", "generic", "llava", "qwen"])
    parser.add_argument("--layers", type=str, default="0,10,20")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    model_key = args.model
    if args.model_config:
        cfg_path = Path(args.model_config)
        if cfg_path.exists() and cfg_path.suffix in {".yaml", ".yml"} and "config" in cfg_path.parts and "model" in cfg_path.parts:
            model_key = cfg_path.stem

    layers = _parse_layers(args.layers)
    prompt = _load_prompt(str(root / args.prompt_path) if not Path(args.prompt_path).exists() else args.prompt_path)

    dataset_dir = Path(args.dataset_dir)
    meta_path = dataset_dir / "meta.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")

    out_dir = Path(args.out_dir)
    tokens_dir = out_dir / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_hydra(model_key)

    with meta_path.open("r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.max_samples is not None:
        records = records[: args.max_samples]

    for rec in tqdm(records, desc="Extract"):
        sample_id = int(rec.get("sample_id"))
        image_path = dataset_dir / rec["image"]
        image = Image.open(image_path).convert("RGB")

        hs = extract_image_hidden_states(
            model=model,
            image=image,
            prompt=prompt,
            layers=layers,
            device=args.device,
            dtype=args.dtype,
        )

        save_path = tokens_dir / f"sample_{sample_id:06d}.npz"
        arrays = {
            "n_image_tokens": np.array(hs.n_image_tokens, dtype=np.int32),
            "image_token_indices": np.asarray(hs.image_token_indices, dtype=np.int32),
            "H_patches": np.array(-1 if hs.h_patches is None else hs.h_patches, dtype=np.int32),
            "W_patches": np.array(-1 if hs.w_patches is None else hs.w_patches, dtype=np.int32),
        }
        for layer, emb in hs.per_layer.items():
            arrays[f"layer_{layer}"] = np.asarray(emb, dtype=np.float32)

        np.savez_compressed(save_path, **arrays)


if __name__ == "__main__":
    main()

