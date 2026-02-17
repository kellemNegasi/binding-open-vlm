from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from models.model import LocalVLModel


@dataclass
class InternVLHFProcessor:
    tokenizer: Any
    image_processor: Any


class InternModel(LocalVLModel):
    """InternVL2 wrapper.

    Notes:
    - `run_batch()` uses LMDeploy (like the Qwen/DeepSeek wrappers).
    - Probing/introspection uses `get_hf_components()` + custom logic in
      `probing/model_introspect.py` (InternVL does not use a Qwen-style processor).
    """

    def __init__(self, prompt_format: Optional[str] = None, **kwargs: Any):
        # Keep Hydra configs compatible with other wrappers that may store metadata.
        self.hf_repo_id = kwargs.pop("hf_repo_id", None)
        super().__init__(**kwargs)
        self.prompt_format = prompt_format

        # Lazy init for inference
        self.pipe = None

        # Lazy init for probing/introspection
        self._hf_model = None
        self._hf_processor = None
        self._hf_components_key = None

    def _prepare_prompt(self, row) -> str:
        prompt = self.format_prompt(row)
        if self.prompt_format:
            prompt = self.prompt_format.format(prompt=prompt)
        return prompt

    def run_batch(self, batch):
        from lmdeploy import pipeline

        if self.pipe is None:
            self.pipe = pipeline(model_path=self.weights_path)

        image_paths = self.get_image_paths(batch)
        prompts = []
        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self._prepare_prompt(row)
            trial_images = self.get_trial_images(row, image_path)
            images = trial_images[0] if len(trial_images) == 1 else trial_images
            prompts.append((prompt_text, images))

        outputs = self.pipe(prompts)
        batch["response"] = [out.text if hasattr(out, "text") else str(out) for out in outputs]
        return batch

    def get_hf_components(self, device: str | None = None, dtype: str | None = None) -> Tuple[Any, InternVLHFProcessor]:
        """Return (hf_model, processor) for probing hidden states.

        The processor is a small container holding:
        - `tokenizer`
        - `image_processor`
        """

        key = (device or "auto", dtype or "auto", self.weights_path)
        if self._hf_model is not None and self._hf_processor is not None and self._hf_components_key == key:
            return self._hf_model, self._hf_processor

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "InternVL probing requires `torch` and `transformers` to be installed."
            ) from e

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

        load_kwargs: dict[str, Any] = dict(trust_remote_code=True, low_cpu_mem_usage=True)
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype

        # For CUDA probing jobs, rely on accelerate to dispatch weights.
        # For CPU-only environments, this still works (slower).
        if device in (None, "auto"):
            load_kwargs["device_map"] = "auto"
        elif device == "cpu":
            load_kwargs["device_map"] = None
        else:
            # Keep the interface consistent with the other wrappers.
            raise ValueError(f"Unsupported device={device!r}. Use auto or cpu.")

        hf_model = AutoModelForCausalLM.from_pretrained(self.weights_path, **load_kwargs)
        hf_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.weights_path, trust_remote_code=True, use_fast=False)
        image_processor = AutoImageProcessor.from_pretrained(self.weights_path, trust_remote_code=True)

        self._hf_model = hf_model
        self._hf_processor = InternVLHFProcessor(tokenizer=tokenizer, image_processor=image_processor)
        self._hf_components_key = key
        return self._hf_model, self._hf_processor
