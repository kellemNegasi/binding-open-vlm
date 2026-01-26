from pathlib import Path
import ast
import torch
from PIL import Image
from transformers import Idefics2ForConditionalGeneration, AutoProcessor
from models.model import LocalVLModel


class Idefics2Model(LocalVLModel):
    def __init__(
        self,
        weights_path,
        prompt_format="Describe the image.",
        task=None,
        batch_size=4,          # increase for speed if GPU allows
        max_tokens=100,
        **kwargs
    ):
        super().__init__(task=task, batch_size=batch_size, **kwargs)

        self.prompt_format = prompt_format or "Describe the image."
        self.max_tokens = max_tokens

        self.processor = AutoProcessor.from_pretrained(weights_path)

        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            weights_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # if available in your env, this can speed up:
            # attn_implementation="flash_attention_2",
        )
        self.model.eval()

    def _abs_path(self, p: str) -> str:
        pp = Path(p)
        if pp.is_absolute():
            return str(pp)
        root = Path(getattr(self, "root_dir", "."))
        return str((root / pp).resolve())

    def _load_images_for_row(self, row):
        # Always return list[Image]
        if "decomposed_paths" in row and row["decomposed_paths"] is not None:
            paths = row["decomposed_paths"]
            if isinstance(paths, str):
                paths = ast.literal_eval(paths)
            return [Image.open(self._abs_path(p)).convert("RGB") for p in paths]

        if "unified_path" in row and isinstance(row["unified_path"], str):
            return [Image.open(self._abs_path(row["unified_path"])).convert("RGB")]

        if "path" in row and isinstance(row["path"], str):
            return [Image.open(self._abs_path(row["path"])).convert("RGB")]

        raise KeyError(f"No image path column found in row keys: {list(row.keys())}")

    def _task_name(self) -> str:
        return str(getattr(self.task, "task_name", "") or getattr(self.task, "name", "") or "").strip().lower()

    def _rmts_subtask(self) -> str:
        return str(getattr(self.task, "subtask", "") or "").strip().lower()

    def _max_new_tokens_for_task(self) -> int:
        t = self._task_name()
        if t in ("conjunctive_search", "disjunctive_search", "disjunctive_search_control"):
            return 6
        if t.startswith("counting"):
            return 6
        if t == "rmts":
            st = self._rmts_subtask()
            if st in ("relations", "features", "full"):
                return 10
            if st == "features2":
                return 256
        if "scene_description" in t:
            return 256
        return self.max_tokens

    def _row_prompt(self, row) -> str:
        for key in ("prompt", "question", "query", "text"):
            if key in row and isinstance(row[key], str) and row[key].strip():
                return row[key].strip()
        return self.prompt_format

    def _format_prompt_for_task(self, base: str) -> str:
        t = self._task_name()
        st = self._rmts_subtask()

        if "scene_description" in t:
            return (
                base
                + "\nReturn ONLY a JSON list like "
                  '[{"color":"red","shape":"circle"}, ...]. '
                  "No extra text."
            )

        if t in ("conjunctive_search", "disjunctive_search", "disjunctive_search_control"):
            return base + "\nAnswer ONLY with [true] or [false]."

        if t.startswith("counting"):
            return base + "\nAnswer ONLY with the number in square brackets, e.g. [5]."

        if t == "rmts":
            if st == "features2":
                return base + "\nReturn ONLY a JSON dict. No extra text."
            if st == "relations":
                return base + "\nAnswer ONLY with [true] or [false]."
            if st == "features":
                return base + "\nAnswer ONLY with the label in brackets, e.g. [red] or [circle]."
            if st == "full":
                return base + "\nAnswer ONLY with the option number in brackets, e.g. [1] or [2]."

        return base

    def run_batch(self, batch):
        images_batch = []
        texts_batch = []

        for _, row in batch.iterrows():
            imgs = self._load_images_for_row(row)  # list[Image]
            base = self._row_prompt(row)
            prompt_text = self._format_prompt_for_task(base)

            # Build IDEFICS2 messages using chat template
            content = []
            for _ in imgs:
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

            images_batch.append(imgs)
            texts_batch.append(text)

        inputs = self.processor(
            text=texts_batch,
            images=images_batch,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        max_new = self._max_new_tokens_for_task()

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
            )

        # Decode ONLY the generated continuation (per-sample)
        decoded = []
        attn = inputs["attention_mask"]
        for i in range(generated.shape[0]):
            prompt_len = int(attn[i].sum().item())
            gen_ids = generated[i, prompt_len:]
            decoded.append(self.processor.decode(gen_ids, skip_special_tokens=True).strip())

        batch["response"] = decoded
        return batch