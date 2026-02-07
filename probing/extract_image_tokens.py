import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from probing.model_introspect import extract_image_hidden_states, load_model_from_hydra


def _parse_layers(s: str) -> list[int] | None:
    if not s:
        return None
    if s.strip().lower() in {"all", "*"}:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _infer_num_hidden_layers(hf_model) -> int:
    cfg = getattr(hf_model, "config", None)
    candidates: list[int] = []

    def _maybe_add(v) -> None:
        try:
            iv = int(v)
        except Exception:
            return
        if iv > 0:
            candidates.append(iv)

    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        _maybe_add(getattr(cfg, attr, None))

    for subcfg_name in ("text_config", "language_config", "llm_config", "model_config"):
        subcfg = getattr(cfg, subcfg_name, None)
        if subcfg is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            _maybe_add(getattr(subcfg, attr, None))

    if not candidates:
        raise RuntimeError(
            "Could not infer number of hidden layers from model config; "
            "pass an explicit --layers list instead of 'all'."
        )
    # Prefer the largest candidate (common when both vision + text configs exist).
    return max(candidates)


def _has_required_layer_keys(npz: np.lib.npyio.NpzFile, layers: list[int]) -> bool:
    files = set(npz.files)
    for layer in layers:
        if f"layer_{int(layer)}" not in files:
            return False
    return True


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
    parser.add_argument("--debug", action="store_true", help="Print debug info for image token indices.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute tokens even if output exists.")
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
    # Resolve "all" layers to an explicit list once so we can validate/resume safely.
    if layers is None:
        hf_model, _ = model.get_hf_components(device=args.device, dtype=args.dtype)
        n_blocks = _infer_num_hidden_layers(hf_model)
        layers_to_extract = list(range(n_blocks))
    else:
        layers_to_extract = layers

    with meta_path.open("r") as f:
        records = [json.loads(line) for line in f if line.strip()]

    if args.max_samples is not None:
        records = records[: args.max_samples]

    for rec in tqdm(records, desc="Extract"):
        sample_id = int(rec.get("sample_id"))
        image_path = dataset_dir / rec["image"]
        image = Image.open(image_path).convert("RGB")

        save_path = tokens_dir / f"sample_{sample_id:06d}.npz"
        if save_path.exists() and not args.overwrite:
            try:
                with np.load(save_path) as z:
                    if _has_required_layer_keys(z, layers_to_extract):
                        continue
            except Exception:
                # Corrupt/incomplete file or incompatible format: recompute.
                pass

        hs = extract_image_hidden_states(
            model=model,
            image=image,
            prompt=prompt,
            layers=layers_to_extract,
            device=args.device,
            dtype=args.dtype,
            debug=args.debug,
        )
        arrays = {
            "n_image_tokens": np.array(hs.n_image_tokens, dtype=np.int32),
            "image_token_indices": np.asarray(hs.image_token_indices, dtype=np.int32),
            "H_patches": np.array(-1 if hs.h_patches is None else hs.h_patches, dtype=np.int32),
            "W_patches": np.array(-1 if hs.w_patches is None else hs.w_patches, dtype=np.int32),
        }
        for layer, emb in hs.per_layer.items():
            arrays[f"layer_{layer}"] = np.asarray(emb, dtype=np.float32)
        if hs.cls_per_layer:
            for layer, emb in hs.cls_per_layer.items():
                arrays[f"cls_{layer}"] = np.asarray(emb, dtype=np.float32)

        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        np.savez_compressed(tmp_path, **arrays)
        tmp_path.replace(save_path)


if __name__ == "__main__":
    main()
