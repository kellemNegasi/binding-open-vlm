import argparse
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--prompt', default="Describe this image in detail.")
    parser.add_argument('--output', default="blip2_output.txt")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")

    model_id = "Salesforce/blip2-flan-t5-xxl"
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()

    inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("BLIP2 response:\n", generated_text)
    Path(args.output).write_text(generated_text)

if __name__ == "__main__":
    main()
