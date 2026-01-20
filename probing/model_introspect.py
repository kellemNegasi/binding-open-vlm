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
    cls_per_layer: Optional[Dict[int, Any]]
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
    return instantiate(cfg.model, task=None, prompt_format="qwen")


def _maybe_prepend_image_token(prompt: str) -> str:
    if "<image>" in prompt or "<|image_pad|>" in prompt or "<|vision_start|>" in prompt:
        return prompt
    return "<image>\n" + prompt


def _single_token_id(tokenizer, token_text: str) -> Optional[int]:
    """Best-effort lookup for special token IDs.

    Some tokenizers do not round-trip special tokens through `encode()` as a single ID.
    Prefer direct vocab / added-vocab lookup, then fall back to `convert_tokens_to_ids`,
    and finally accept `encode()` only when it returns a single ID.
    """
    # 1) Direct vocab lookup (fast + reliable for special tokens).
    try:
        vocab = tokenizer.get_vocab()
        if token_text in vocab:
            return int(vocab[token_text])
    except Exception:
        pass

    # 2) Added vocab (common for additional special tokens).
    try:
        added = tokenizer.get_added_vocab()
        if token_text in added:
            return int(added[token_text])
    except Exception:
        pass

    # 3) convert_tokens_to_ids (may return unk_token_id if missing).
    try:
        tid = tokenizer.convert_tokens_to_ids(token_text)
        if tid is not None:
            unk = getattr(tokenizer, "unk_token_id", None)
            if unk is None or int(tid) != int(unk):
                return int(tid)
    except Exception:
        pass

    # 4) Last resort: accept encode() only if it produces exactly one token id.
    try:
        ids = tokenizer.encode(token_text, add_special_tokens=False)
        if isinstance(ids, list) and len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass

    return None


def _infer_image_token_indices(tokenizer, input_ids) -> Any:
    import torch

    # Qwen-VL style (common): a vision span demarcated by vision_start / vision_end,
    # or repeated pad tokens. Keep candidates extensible since different releases
    # may register these as "added" tokens rather than normal vocab entries.
    candidates_pad = ["<|image_pad|>", "<|vision_pad|>"]
    candidates_vstart = ["<|vision_start|>"]
    candidates_vend = ["<|vision_end|>"]

    def _discover_from_special_tokens(needle: str) -> list[str]:
        toks: list[str] = []
        for t in getattr(tokenizer, "additional_special_tokens", []) or []:
            if isinstance(t, str) and needle in t:
                toks.append(t)
        try:
            special_map = tokenizer.special_tokens_map_extended
        except Exception:
            special_map = {}
        for v in (special_map or {}).values():
            if isinstance(v, str):
                v = [v]
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, str) and needle in t:
                        toks.append(t)
        # preserve order, de-dupe
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    # If the canonical candidates aren't registered as single tokens, fall back to any special tokens
    # that contain the expected substrings (e.g., different delimiter styles across versions).
    candidates_pad = candidates_pad + _discover_from_special_tokens("image_pad")
    candidates_pad = candidates_pad + _discover_from_special_tokens("vision_pad")
    candidates_vstart = candidates_vstart + _discover_from_special_tokens("vision_start")
    candidates_vend = candidates_vend + _discover_from_special_tokens("vision_end")

    pad_id = next((tid for t in candidates_pad if (tid := _single_token_id(tokenizer, t)) is not None), None)
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

    if pad_id is not None:
        idx = torch.where(input_ids[0] == pad_id)[0]
        if len(idx) > 0:
            return idx

    # Some tokenizers expose a dedicated image/vision token id; try those as a last resort.
    attr_ids = [
        "image_token_id",
        "vision_token_id",
        "image_pad_token_id",
        "vision_pad_token_id",
    ]
    for name in attr_ids:
        tid = getattr(tokenizer, name, None)
        if tid is not None:
            idx = torch.where(input_ids[0] == int(tid))[0]
            if len(idx) > 0:
                return idx

    attr_tokens = [
        "image_token",
        "vision_token",
        "image_pad_token",
        "vision_pad_token",
    ]
    for name in attr_tokens:
        tok = getattr(tokenizer, name, None)
        if isinstance(tok, str):
            tid = _single_token_id(tokenizer, tok)
            if tid is not None:
                idx = torch.where(input_ids[0] == tid)[0]
                if len(idx) > 0:
                    return idx

    try:
        specials = list(getattr(tokenizer, "additional_special_tokens", []) or [])
        specials = [t for t in specials if isinstance(t, str)]
        hint = [t for t in specials if any(s in t for s in ("vision", "image"))][:20]
    except Exception:
        hint = []

    raise ValueError(
        "Unable to infer image token positions. "
        "For Qwen-VL models, ensure the processor inserts repeated `<|image_pad|>` tokens, "
        "or update `_infer_image_token_indices()` for this wrapper. "
        f"Tokenizer additional_special_tokens (vision/image-related, first 20): {hint}"
    )


def _infer_vision_start_index(tokenizer, input_ids) -> Optional[int]:
    import torch

    candidates_vstart = ["<|vision_start|>"]

    def _discover_from_special_tokens(needle: str) -> list[str]:
        toks: list[str] = []
        for t in getattr(tokenizer, "additional_special_tokens", []) or []:
            if isinstance(t, str) and needle in t:
                toks.append(t)
        try:
            special_map = tokenizer.special_tokens_map_extended
        except Exception:
            special_map = {}
        for v in (special_map or {}).values():
            if isinstance(v, str):
                v = [v]
            if isinstance(v, (list, tuple)):
                for t in v:
                    if isinstance(t, str) and needle in t:
                        toks.append(t)
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    candidates_vstart = candidates_vstart + _discover_from_special_tokens("vision_start")
    vstart_id = next((tid for t in candidates_vstart if (tid := _single_token_id(tokenizer, t)) is not None), None)
    if vstart_id is None:
        return None
    idx = torch.where(input_ids[0] == vstart_id)[0]
    if len(idx) == 0:
        return None
    return int(idx[0].item())


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

    # prompt = _maybe_prepend_image_token(prompt)
    # inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)

    # Build multimodal chat messages (Qwen-VL expects this)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Qwen-style: chat template is usually on the tokenizer
    if hasattr(tokenizer, "apply_chat_template"):
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        # fallback (less reliable for Qwen-VL)
        chat_text = _maybe_prepend_image_token(prompt)

    inputs = processor(
        text=[chat_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )


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
    
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    print("vision_start in input_ids?", "<|vision_start|>" in tokens)
    print("vision_end in input_ids?", "<|vision_end|>" in tokens)
    print("image_pad count:", sum(t == "<|image_pad|>" for t in tokens))
    print("vision_pad count:", sum(t == "<|vision_pad|>" for t in tokens))
    print("First 120 tokens:", tokens[:120])
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
    cls_per_layer: Optional[Dict[int, Any]] = None
    cls_index = _infer_vision_start_index(tokenizer, input_ids)
    if cls_index is not None:
        cls_per_layer = {}
    n_blocks = len(hidden_states) - 1  # hidden_states[0] is embeddings
    for layer in layers:
        if layer < 0:
            layer = n_blocks + layer
        if layer < 0 or layer >= n_blocks:
            raise ValueError(f"Requested layer {layer} out of range [0, {n_blocks - 1}]")
        hs = hidden_states[layer + 1]  # [B, seq, d]
        per_layer[int(layer)] = hs[0, image_token_indices, :].detach().cpu()
        if cls_per_layer is not None:
            cls_per_layer[int(layer)] = hs[0, cls_index, :].detach().cpu()

    return ImageHiddenStates(
        per_layer=per_layer,
        cls_per_layer=cls_per_layer,
        n_image_tokens=n_image_tokens,
        image_token_indices=image_token_indices.detach().cpu(),
        h_patches=h_patches,
        w_patches=w_patches,
    )
