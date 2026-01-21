from models.model import LocalVLModel
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

class Gemma3Model(LocalVLModel):
    def __init__(self, prompt_format=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_format = prompt_format
        self.pipe = None
        self.gen_config = GenerationConfig(
            max_new_tokens=min(self.max_tokens, 16),  # clamp hard for counting
            temperature=0.0
        )

    def _prepare_prompt(self, row):
        prompt = self.format_prompt(row)
        if self.prompt_format:
            prompt = self.prompt_format.format(prompt=prompt)
        return prompt

    def run_batch(self, batch):
        if self.pipe is None:
            # try turbomind tp=2, but if unsupported you'll still fall back
            # self.pipe = pipeline(
            #     model_path=self.weights_path,
            #     backend_config=TurbomindEngineConfig(tp=2)
            # )
            self.pipe = pipeline(
            model_path="google/gemma-3-27b-it",
            backend_config=TurbomindEngineConfig(tp=2),
            )


        image_paths = self.get_image_paths(batch)
        prompts = []
        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self._prepare_prompt(row)
            trial_images = self.get_trial_images(row, image_path)  # PIL Images already
            images = trial_images[0] if len(trial_images) == 1 else trial_images
            prompts.append((prompt_text, images))

        generated = self.pipe(prompts, gen_config=self.gen_config)
        batch["response"] = [out.text if hasattr(out, "text") else str(out) for out in generated]
        return batch
