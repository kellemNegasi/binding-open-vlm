from models.model import LocalVLModel
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class Blip2Model(LocalVLModel):
    def __init__(self, prompt_format=None, **kwargs):
        super().__init__(**kwargs)
        if prompt_format:
            self.task.prompt = prompt_format.format(prompt=self.task.prompt)
        self.processor = Blip2Processor.from_pretrained(self.weights_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.weights_path,
            device_map='auto',
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

    def run_batch(self, batch):
        image_paths = self.get_image_paths(batch)
        texts, images = [], []
        for (_, row), path in zip(batch.iterrows(), image_paths):
            texts.append(self.format_prompt(row))
            images.append(self.compose_images(self.get_trial_images(row, path)))
        inputs = self.processor(images=images, text=texts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        batch['response'] = outputs
        return batch
