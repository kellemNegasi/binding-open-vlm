from models.model import LocalVLModel
from lmdeploy import pipeline
import subprocess as sp

class QwenModel(LocalVLModel):
        
    def __init__(self, prompt_format, **kwargs):
        super().__init__(**kwargs)
        # Lazily initialize lmdeploy pipeline so probing/introspection can instantiate
        # the wrapper without requiring the lmdeploy runtime/device to be available.
        self.pipe = None
        self._hf_model = None
        self._hf_processor = None
        self._hf_components_key = None
        
    def run_batch(self, batch):
        if self.pipe is None:
            self.pipe = pipeline(model_path=self.weights_path)
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
            from transformers import activations as hf_activations
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Hidden-state introspection for Qwen requires `torch` and `transformers`."
            ) from e
        # Compatibility shim: AWQ expects PytorchGELUTanh, removed in newer transformers.
        if not hasattr(hf_activations, "PytorchGELUTanh"):
            class PytorchGELUTanh(torch.nn.Module):
                def forward(self, x):
                    return torch.nn.functional.gelu(x, approximate="tanh")

            hf_activations.PytorchGELUTanh = PytorchGELUTanh

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

        # Device handling:
        # - default/auto: let HF/Accelerate decide (device_map='auto')
        # - xpu: load on CPU then move to XPU (device_map=None) to avoid CUDA assumptions
        load_kwargs = dict(trust_remote_code=True)
        if torch_dtype != "auto":
            load_kwargs["torch_dtype"] = torch_dtype

        if device in (None, "auto"):
            load_kwargs["device_map"] = "auto"
        elif device == "xpu":
            # Some environments require IPEX for XPU; attempt import but don't hard-require it here.
            try:  # pragma: no cover
                import intel_extension_for_pytorch  # noqa: F401
            except Exception:
                pass
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError(
                    "Requested device=xpu but torch.xpu is not available. "
                    "Install/enable the XPU backend (e.g., Intel Extension for PyTorch) and retry."
                )
            load_kwargs["device_map"] = None
        else:
            raise ValueError(f"Unsupported device={device!r}. Use auto or xpu.")

        hf_model = AutoModelForVision2Seq.from_pretrained(
            self.weights_path,
            **load_kwargs,
        )
        if device == "xpu":
            hf_model.to("xpu")
        hf_model.eval()

        processor = AutoProcessor.from_pretrained(self.weights_path, trust_remote_code=True)

        self._hf_model = hf_model
        self._hf_processor = processor
        self._hf_components_key = key
        return self._hf_model, self._hf_processor
