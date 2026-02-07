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


def _is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    r = int(round(n**0.5))
    return r * r == n


def _sqrt_int(n: int) -> int | None:
    if not _is_perfect_square(n):
        return None
    return int(round(n**0.5))


def _internvl2_pixel_values(
    image: Image.Image,
    *,
    device,
    dtype,
    size: int = 448,
) -> "torch.Tensor":
    import torch

    img = image.convert("RGB").resize((size, size), resample=Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0  # HWC, [0,1]
    # CHW
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype)[:, None, None]
    t = (t - mean) / std
    t = t.unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def _pixel_values_from_processor(processor, image: Image.Image):
    """Best-effort creation of `pixel_values` from an HF processor.

    Some processors require `text` even if we only need images; others can process images alone.
    """
    try:
        from transformers import PreTrainedTokenizerBase

        if isinstance(processor, PreTrainedTokenizerBase):
            # Tokenizers accept arbitrary kwargs and will warn/ignore `images=...`.
            return None
    except Exception:
        pass

    # 1) Images-only path.
    try:
        out = processor(images=[image], return_tensors="pt")
        pv = out.get("pixel_values", None)
        if pv is not None:
            return pv
    except Exception:
        pass

    # 2) Multimodal path: blank text + image.
    try:
        out = processor(text=[""], images=[image], return_tensors="pt", padding=True)
        pv = out.get("pixel_values", None)
        if pv is not None:
            return pv
    except Exception:
        pass

    return None


def load_model_from_hydra(model_name: str):
    """Instantiate a repo model wrapper via Hydra config (like `run_vlm.py`)."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    # `initialize()` requires a *relative* config_path; use initialize_config_dir for an absolute path.
    with initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        cfg = compose(config_name="run", overrides=[f"model={model_name}", "task=default", "paths=default"])
    # Many wrappers require `task` at construction time; allow `task=None` for probing use-cases.
    return instantiate(cfg.model, task=None, prompt_format="qwen")


class HfRepoModel:
    """Minimal wrapper that exposes `get_hf_components()` for probing/introspection.

    This allows probing to run directly from a Hugging Face repo id (or local model folder)
    without requiring a Hydra config/model wrapper.
    """

    def __init__(self, weights_path: str, hf_repo_id: str | None = None):
        self.weights_path = weights_path
        self.hf_repo_id = hf_repo_id
        self._hf_model = None
        self._hf_processor = None
        self._hf_components_key = None

    def get_hf_components(self, device: str | None = None, dtype: str | None = None):
        key = (device or "auto", dtype or "auto", self.weights_path)
        if self._hf_model is not None and self._hf_processor is not None and self._hf_components_key == key:
            return self._hf_model, self._hf_processor

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except Exception as e:  # pragma: no cover
            raise ImportError("Hidden-state introspection requires `torch` and `transformers`.") from e

        if device is not None:
            device = device.lower()

        dtype_map = {
            None: "auto",
            "auto": "auto",
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get((dtype or "auto").lower(), "auto")

        load_kwargs = dict(trust_remote_code=True)
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype

        if device in (None, "auto"):
            load_kwargs["device_map"] = "auto"
        elif device == "xpu":
            load_kwargs["device_map"] = None
        else:
            raise ValueError(f"Unsupported device={device!r}. Use auto or xpu.")

        hf_model = AutoModel.from_pretrained(self.weights_path, **load_kwargs)
        if device == "xpu":
            hf_model.to("xpu")
        hf_model.eval()

        processor = AutoProcessor.from_pretrained(self.weights_path, trust_remote_code=True)

        self._hf_model = hf_model
        self._hf_processor = processor
        self._hf_components_key = key
        return self._hf_model, self._hf_processor


def load_model(model_name: str):
    """Load a probing-compatible model wrapper.

    - If `model_name` looks like a Hugging Face repo id (contains '/'), load directly via transformers.
      If a matching local folder exists under `model-weights/<repo_name>`, prefer that to avoid downloads.
    - Otherwise, treat `model_name` as a Hydra model key under `config/model/*.yaml`.
    """
    if "/" in model_name:
        repo_name = model_name.split("/")[-1]
        local_dir = root / "model-weights" / repo_name
        weights_path = str(local_dir) if local_dir.exists() else model_name
        return HfRepoModel(weights_path=weights_path, hf_repo_id=model_name)
    return load_model_from_hydra(model_name)


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
    candidates_pad = ["<|image_pad|>", "<|vision_pad|>", "<image>"]
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

    # More general fallback: pick any special token id that appears repeatedly and looks vision-related.
    try:
        candidate_tokens = []
        for needle in ("image", "vision", "img"):
            candidate_tokens.extend(_discover_from_special_tokens(needle))
        # preserve order, de-dupe
        seen = set()
        candidate_tokens = [t for t in candidate_tokens if not (t in seen or seen.add(t))]
        candidate_ids = []
        for t in candidate_tokens:
            tid = _single_token_id(tokenizer, t)
            if tid is not None:
                candidate_ids.append(int(tid))

        best = None  # (count, tid)
        for tid in candidate_ids:
            count = int((input_ids[0] == tid).sum().item())
            if count <= 1:
                continue
            if best is None or count > best[0]:
                best = (count, tid)
        if best is not None:
            _count, tid = best
            idx = torch.where(input_ids[0] == tid)[0]
            if len(idx) > 0:
                return idx
    except Exception:
        pass

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
    layers: Optional[List[int]],
    device: str | None = None,
    dtype: str | None = None,
    debug: bool = False,
) -> ImageHiddenStates:
    """Extract per-layer hidden states for image tokens only.

    Supports:
      - Tokenizer-based multimodal models (Qwen-VL style) via `processor.tokenizer` to locate image tokens.
      - Vision-only extraction when the processor has no tokenizer but the HF model exposes a vision encoder
        (e.g. InternVL2), in which case we extract patch-token hidden states from the vision transformer.
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
        # Vision-only fallback (used for InternVL2-style models).
        vision = None
        for attr in ("vision_model", "vision_encoder", "visual", "vision_tower"):
            cand = getattr(hf_model, attr, None)
            if cand is not None:
                vision = cand
                break
        if vision is None:
            raise NotImplementedError(
                "Processor does not expose a tokenizer and no vision encoder was found on the model; "
                "cannot identify image tokens."
            )

        # Use the processor to build pixel_values if possible (preferred), otherwise fall back to
        # a known-good InternVL2 preprocessing recipe.
        try:
            vision_dtype = next(vision.parameters()).dtype
        except Exception:  # pragma: no cover
            vision_dtype = torch.float32

        pixel_values = _pixel_values_from_processor(processor, image)
        if pixel_values is None:
            repo_hint = str(getattr(model, "hf_repo_id", "") or getattr(model, "weights_path", "") or "").lower()
            if "internvl" in repo_hint:
                pixel_values = _internvl2_pixel_values(
                    image,
                    device=target_device,
                    dtype=vision_dtype,
                    size=448,
                )
            else:
                raise RuntimeError(
                    "Processor did not produce `pixel_values`; cannot run vision encoder for hidden states. "
                    "If this is an InternVL2 model, ensure the repo id contains 'InternVL' or add a custom preprocessor."
                )

        if torch.is_tensor(pixel_values):
            pixel_values = pixel_values.to(device=target_device, dtype=vision_dtype)

        # Run the vision transformer and collect hidden states.
        with torch.no_grad():
            try:
                v_out = vision(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            except TypeError:
                v_out = vision(pixel_values, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(v_out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Vision encoder did not return `hidden_states`. Ensure `output_hidden_states=True` is supported."
            )

        # Most HF vision transformers return hidden_states as (embeddings, layer1, ..., layerN).
        n_blocks = len(hidden_states) - 1
        if n_blocks <= 0:
            raise RuntimeError("Vision encoder returned too few hidden state tensors.")
        if not layers or len(layers) == 0:
            layers = list(range(n_blocks))

        # Determine whether a CLS token exists (common): seq_len = 1 + (h*w).
        hs0 = hidden_states[1]
        if not (hasattr(hs0, "shape") and len(hs0.shape) == 3):
            raise RuntimeError("Unexpected hidden state shape from vision encoder (expected [B, seq, d]).")
        seq_len = int(hs0.shape[1])
        if _is_perfect_square(seq_len - 1):
            # Assume first token is CLS.
            idx = torch.arange(1, seq_len, device=target_device, dtype=torch.long)
            n_image_tokens = seq_len - 1
            side = _sqrt_int(n_image_tokens)
        else:
            idx = torch.arange(0, seq_len, device=target_device, dtype=torch.long)
            n_image_tokens = seq_len
            side = _sqrt_int(n_image_tokens)

        per_layer: Dict[int, Any] = {}
        for layer in layers:
            if layer < 0:
                layer = n_blocks + layer
            if layer < 0 or layer >= n_blocks:
                raise ValueError(f"Requested layer {layer} out of range [0, {n_blocks - 1}]")
            hs = hidden_states[layer + 1]  # [B, seq, d]
            per_layer[int(layer)] = hs[0, idx, :].detach().cpu()

        return ImageHiddenStates(
            per_layer=per_layer,
            cls_per_layer=None,
            n_image_tokens=int(n_image_tokens),
            image_token_indices=idx.detach().cpu(),
            h_patches=None if side is None else int(side),
            w_patches=None if side is None else int(side),
        )

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
    
    if debug:
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
    if not layers or len(layers) == 0:
        layers = list(range(n_blocks))

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
