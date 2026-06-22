"""Multimodal input for Gemma 4 via its reserved media tokens.

google/gemma-4-12b-it is Gemma4UnifiedForConditionalGeneration — a unified model
with both a vision and an audio tower. In the chat template, a content part of
type image/audio/video renders to the reserved placeholder <|image|> / <|audio|>
/ <|video|> (verified in probe_chat_template.py). The processor expands that
placeholder into the media embeddings and the matching pixel/audio tensors.

This module builds those multimodal messages and runs generation through the
processor + conditional-generation model. The text-only agent loop in gemma4.py
deliberately stays on the lighter AutoModelForCausalLM path; this is the heavier
media path, loaded only when you actually need it.

Demo:  python multimodal.py    (synthesizes a test image and asks about it)
"""

from tokens import AUDIO, IMAGE, VIDEO

MODEL_ID = "google/gemma-4-12b-it"

# media kind -> reserved placeholder token, for reference/inspection.
PLACEHOLDER = {"image": IMAGE, "audio": AUDIO, "video": VIDEO}


def media_message(prompt, image=None, audio=None, video=None):
    """Build a single user message mixing media + text.

    image/audio/video may be a PIL image, a path, or a URL (whatever the
    processor accepts). Media parts come first, then the text prompt — matching
    how the template emits the <|image|>/<|audio|>/<|video|> placeholder ahead of
    the text.
    """
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if audio is not None:
        content.append({"type": "audio", "audio": audio})
    if video is not None:
        content.append({"type": "video", "video": video})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class MultimodalEngine:
    """Loads the processor + conditional-generation model for media input."""

    def __init__(self, model_id=MODEL_ID):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )

    def generate(self, messages, max_new_tokens=256):
        """Run the model on processor-built multimodal messages; return text."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

    def ask_image(self, image, prompt, **kw):
        return self.generate(media_message(prompt, image=image), **kw)


def _make_test_image():
    """Synthesize a deterministic test image: a red circle on white."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((52, 52, 172, 172), fill="red")
    return img


def main():
    print("Reserved media placeholders:", PLACEHOLDER)
    img = _make_test_image()
    img.save("test_image.png")
    print("Wrote test_image.png (a red circle on white).")

    engine = MultimodalEngine()
    answer = engine.ask_image(img, "What single shape is in this image, and what color is it? Answer in one short sentence.")
    print("\n[Model sees the image]:")
    print(answer)


if __name__ == "__main__":
    main()
