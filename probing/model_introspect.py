from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


@dataclass
class ImageHiddenStates:
    per_layer: Dict[int, torch.Tensor]
    n_image_tokens: int
    h_patches: int
    w_patches: int


class InternVL2Introspector:
    """
    Vision-only introspection for OpenGVLab/InternVL2-26B (HF).
    """

    def __init__(
        self,
        model_name: str = "OpenGVLab/InternVL2-26B",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device("cuda")

        # ✅ CORRECT loader for HF checkpoint
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        # Vision encoder lives here
        self.vision_model = self.model.vision_model

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def extract(
        self,
        image: Image.Image,
        prompt: str,   # kept for pipeline compatibility
        layers: List[int],
    ) -> ImageHiddenStates:

        image = image.convert("RGB")
        pixel_values = (
            self.transform(image)
            .unsqueeze(0)
            .to(self.device)
            .to(dtype=torch.float16)
        )

        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        n_blocks = len(hidden_states)

        per_layer: Dict[int, torch.Tensor] = {}

        for layer in layers:
            if layer < 0:
                layer = n_blocks + layer
            if not (0 <= layer < n_blocks):
                raise ValueError(f"Layer {layer} out of range")

            per_layer[layer] = hidden_states[layer][0].cpu()

        num_patches = per_layer[layers[0]].shape[0]
        h = w = int(num_patches ** 0.5)

        return ImageHiddenStates(
            per_layer=per_layer,
            n_image_tokens=num_patches,
            h_patches=h,
            w_patches=w,
        )
