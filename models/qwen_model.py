from models.model import LocalVLModel
from lmdeploy import pipeline
import subprocess as sp

class QwenModel(LocalVLModel):
        
    def __init__(self, prompt_format, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipeline(model_path=self.weights_path)
        
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
