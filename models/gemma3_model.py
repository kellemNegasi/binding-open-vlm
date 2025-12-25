from models.model import LocalVLModel
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl import load_image

class Gemma3Model(LocalVLModel):
    def __init__(self, prompt_format=None, **kwargs):
        super().__init__(**kwargs)
        self.prompt_format = prompt_format

        # IMPORTANT for 2x A100: use tp=2, otherwise you often OOM on 27B
        self.pipe = pipeline(
            model_path=self.weights_path,
            backend_config=TurbomindEngineConfig(tp=2)
        )
        self.gen_config = GenerationConfig(max_new_tokens=self.max_tokens)

    def _prepare_prompt(self, row):
        prompt = self.format_prompt(row)
        if self.prompt_format:
            prompt = self.prompt_format.format(prompt=prompt)
        return prompt

    def run_batch(self, batch):
        image_paths = self.get_image_paths(batch)
        prompts = []

        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self._prepare_prompt(row)
            trial_images = self.get_trial_images(row, image_path)

            # get_trial_images returns a list; LMDeploy accepts either single image or list-of-images
            images = [load_image(img) if isinstance(img, str) else img for img in trial_images]
            images = images[0] if len(images) == 1 else images

            # DO NOT add IMAGE_TOKEN for Gemma unless you truly need custom placement
            prompts.append((prompt_text, images))

        generated = self.pipe(prompts, gen_config=self.gen_config)

        batch["response"] = [
            out.text if hasattr(out, "text") else str(out)
            for out in generated
        ]
        return batch

