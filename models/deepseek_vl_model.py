from __future__ import annotations

from models.model import LocalVLModel
from lmdeploy import pipeline, GenerationConfig
from typing import Any


class DeepSeekVL2Model(LocalVLModel):
    """DeepSeek-VL2 wrapper using LMDeploy pipeline."""

    def __init__(self, prompt_format: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.prompt_format = prompt_format

        # Lazy init (consistent with your Qwen wrapper)
        self.pipe = None

        # Keep decoding small by default; override via Hydra: model.max_tokens=...
        self.gen_config = GenerationConfig(
            max_new_tokens=min(getattr(self, "max_tokens", 64), 64),
            temperature=0.0,
        )

    def _prepare_prompt(self, row) -> str:
        prompt = self.format_prompt(row)
        if self.prompt_format:
            prompt = self.prompt_format.format(prompt=prompt)
        return prompt

    def run_batch(self, batch):
        # Create pipeline once per job
        if self.pipe is None:
            # NOTE: If TurboMind is unavailable on your cluster (libcuda issue),
            # LMDeploy will fall back to PyTorch engine automatically.
            self.pipe = pipeline(model_path=self.weights_path)

        image_paths = self.get_image_paths(batch)
        prompts = []

        # Build LMDeploy (prompt, image(s)) tuples
        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self._prepare_prompt(row)
            trial_images = self.get_trial_images(row, image_path)  # PIL.Image objects

            images = trial_images[0] if len(trial_images) == 1 else trial_images
            prompts.append((prompt_text, images))

        outputs = self.pipe(prompts, gen_config=self.gen_config)

        batch["response"] = [
            out.text if hasattr(out, "text") else str(out)
            for out in outputs
        ]
        return batch
