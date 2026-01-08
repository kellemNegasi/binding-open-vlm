import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--output", default="idefics2_output.txt")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    image = Image.open(image_path).convert("RGB")

    model_id = "HuggingFaceM4/idefics2-8b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # ✅ Image is a placeholder in the chat; the real PIL image is passed separately to processor(...)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # ✅ IMPORTANT: use keyword args (text=..., images=...)
    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("IDEFICS2 response:\n", out)
    Path(args.output).write_text(out)

if __name__ == "__main__":
    main()
