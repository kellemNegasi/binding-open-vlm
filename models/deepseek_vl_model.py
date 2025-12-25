from models.model import LocalVLModel
from lmdeploy import pipeline
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

class DeepseekVL2Model(LocalVLModel):
    def __init__(self, prompt_format=None, **kwargs):
        super().__init__(**kwargs)
        self.pipe = pipeline(model_path=self.weights_path)  # local folder works

        # optional prompt wrapper
        self.prompt_format = prompt_format

    def run_batch(self, batch):
        image_paths = self.get_image_paths(batch)
        prompts = []

        for (_, row), image_path in zip(batch.iterrows(), image_paths):
            prompt_text = self.format_prompt(row)
            if self.prompt_format:
                prompt_text = self.prompt_format.format(prompt=prompt_text)

            # your framework may produce “trial images”; adapt:
            # - if you have ONE image per example, use that
            # - if you have MULTIPLE images, pass a list
            trial_images = self.get_trial_images(row, image_path)

            if isinstance(trial_images, (list, tuple)):
                images = [load_image(p) if isinstance(p, str) else p for p in trial_images]
                # safest: one IMAGE_TOKEN per image
                prompt = (IMAGE_TOKEN * len(images)) + prompt_text
                prompts.append((prompt, images))
            else:
                image = load_image(trial_images) if isinstance(trial_images, str) else trial_images
                prompt = f"{IMAGE_TOKEN}{prompt_text}"
                prompts.append((prompt, image))

        outputs = self.pipe(prompts)

        responses = []
        for out in outputs:
            responses.append(out.text if hasattr(out, "text") else str(out))

        batch["response"] = responses
        return batch
