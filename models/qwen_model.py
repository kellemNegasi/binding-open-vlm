from models.model import LocalVLModel
from lmdeploy import pipeline
import subprocess as sp

class QwenModel(LocalVLModel):
        
    def __init__(self, prompt_format, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipeline(model_path=self.weights_path)
        self._hf_model = None
        self._hf_processor = None
        self._hf_components_key = None
        
    def run_batch(self, batch):
        image_paths = self.get_image_paths(batch)
        prompts = []
        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self.format_prompt(row)
            trial_images = self.get_trial_images(row, image_path)
            prompts.append((prompt_text, trial_images))
        generated_texts = self.pipe(prompts)
        responses = []
        for output in generated_texts:
            if hasattr(output, 'text'):
                responses.append(output.text)
            else:
                responses.append(str(output))
        batch['response'] = responses
        return batch

    def get_gpu_memory(self):
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values

    def get_hf_components(self, device: str | None = None, dtype: str | None = None):
        """Return (hf_model, hf_processor) for probing hidden states.

        This is optional functionality and does not affect normal inference via lmdeploy.
        """
        key = (device or "auto", dtype or "auto", self.weights_path)
        if self._hf_model is not None and self._hf_processor is not None and self._hf_components_key == key:
            return self._hf_model, self._hf_processor

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Hidden-state introspection for Qwen requires `torch` and `transformers`."
            ) from e

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

        # For large models, let HF place weights automatically (device_map='auto').
        device_map = "auto" if device in (None, "auto") else "auto"
        hf_model = AutoModelForVision2Seq.from_pretrained(
            self.weights_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        hf_model.eval()

        processor = AutoProcessor.from_pretrained(self.weights_path, trust_remote_code=True)

        self._hf_model = hf_model
        self._hf_processor = processor
        self._hf_components_key = key
        return self._hf_model, self._hf_processor
