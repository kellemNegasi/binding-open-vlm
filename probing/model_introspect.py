from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import pyrootutils

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@dataclass
class ImageHiddenStates:
    per_layer: Dict[int, Any]
    n_image_tokens: int
    image_token_indices: Any
    h_patches: Optional[int] = None
    w_patches: Optional[int] = None


def load_model_from_hydra(model_name: str):
    """Instantiate a repo model wrapper via Hydra config (like `run_vlm.py`)."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    # `initialize()` requires a *relative* config_path; use initialize_config_dir for an absolute path.
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        cfg = compose(config_name="run", overrides=[f"model={model_name}", "task=default", "paths=default"])
    # Many wrappers require `task` at construction time; allow `task=None` for probing use-cases.
    return instantiate(cfg.model, task=None)


def _maybe_prepend_image_token(prompt: str) -> str:
    if "<image>" in prompt or "<|image_pad|>" in prompt or "<|vision_start|>" in prompt:
        return prompt
    return "<image>\n" + prompt


def _single_token_id(tokenizer, token_text: str) -> Optional[int]:
    try:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
    except Exception:
        return None
    if isinstance(ids, list) and len(ids) == 1:
        return int(ids[0])
    return None


def _infer_image_token_indices(tokenizer, input_ids) -> Any:
    import torch

    candidates_pad = ["<|image_pad|>"]
    candidates_vstart = ["<|vision_start|>", "<|vision_start|>"]
    candidates_vend = ["<|vision_end|>", "<|vision_end|>"]

    image_pad_id = next((tid for t in candidates_pad if (tid := _single_token_id(tokenizer, t)) is not None), None)
    vstart_id = next((tid for t in candidates_vstart if (tid := _single_token_id(tokenizer, t)) is not None), None)
    vend_id = next((tid for t in candidates_vend if (tid := _single_token_id(tokenizer, t)) is not None), None)

    if vstart_id is not None and vend_id is not None:
        start_pos = torch.where(input_ids[0] == vstart_id)[0]
        end_pos = torch.where(input_ids[0] == vend_id)[0]
        if len(start_pos) > 0 and len(end_pos) > 0:
            start = int(start_pos[0].item())
            end = int(end_pos[-1].item())
            if end > start + 1:
                return torch.arange(start + 1, end, device=input_ids.device, dtype=torch.long)

    if image_pad_id is not None:
        idx = torch.where(input_ids[0] == image_pad_id)[0]
        if len(idx) > 0:
            return idx

    raise ValueError(
        "Unable to infer image token positions. "
        "For Qwen-VL models, ensure the processor inserts repeated `<|image_pad|>` tokens, "
        "or update `_infer_image_token_indices()` for this wrapper."
    )


def extract_image_hidden_states(
    model,
    image: Image.Image,
    prompt: str,
    layers: List[int],
    device: str | None = None,
    dtype: str | None = None,
) -> ImageHiddenStates:
    """Extract per-layer hidden states for image tokens only.

    Currently supports Qwen wrappers via `models/qwen_model.py:QwenModel.get_hf_components()`.
    """
    if not hasattr(model, "get_hf_components"):
        raise NotImplementedError(
            f"Hidden-state extraction is not implemented for model wrapper {type(model).__name__}. "
            "Add a `get_hf_components()` method (or equivalent) that returns an HF model+processor, "
            "then extend `probing/model_introspect.py`."
        )

    import torch

    hf_model, processor = model.get_hf_components(device=device, dtype=dtype)
    try:
        target_device = next(hf_model.parameters()).device
    except StopIteration:  # pragma: no cover
        target_device = getattr(hf_model, "device", None) or torch.device("cpu")
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise NotImplementedError("Processor does not expose a tokenizer; cannot identify image tokens.")

    prompt = _maybe_prepend_image_token(prompt)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(target_device)

    with torch.no_grad():
        outputs = hf_model(**inputs, output_hidden_states=True, return_dict=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Model did not return `hidden_states`. Ensure `output_hidden_states=True` is supported.")

    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        raise RuntimeError("Processor did not produce `input_ids`; cannot infer image token positions.")

    image_token_indices = _infer_image_token_indices(tokenizer, input_ids)
    n_image_tokens = int(image_token_indices.shape[0])

    h_patches = None
    w_patches = None
    grid = inputs.get("image_grid_thw", None)
    if grid is not None:
        try:
            grid = grid[0].detach().cpu().tolist()
            # Expected: [t, h, w] for single image.
            if isinstance(grid, (list, tuple)) and len(grid) == 3:
                h_patches = int(grid[1])
                w_patches = int(grid[2])
        except Exception:
            pass

    per_layer: Dict[int, Any] = {}
    n_blocks = len(hidden_states) - 1  # hidden_states[0] is embeddings
    for layer in layers:
        if layer < 0:
            layer = n_blocks + layer
        if layer < 0 or layer >= n_blocks:
            raise ValueError(f"Requested layer {layer} out of range [0, {n_blocks - 1}]")
        hs = hidden_states[layer + 1]  # [B, seq, d]
        per_layer[int(layer)] = hs[0, image_token_indices, :].detach().cpu()

    return ImageHiddenStates(
        per_layer=per_layer,
        n_image_tokens=n_image_tokens,
        image_token_indices=image_token_indices.detach().cpu(),
        h_patches=h_patches,
        w_patches=w_patches,
    )
