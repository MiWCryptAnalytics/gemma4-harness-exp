"""UnifiedEngine — one Gemma 4 model instance for BOTH text tool-calling and vision.

The agent loop (gemma4.run_agent) and the image tool (image_tool.create_image)
must share a single model or they'd need ~2x VRAM. The unified model
(Gemma4UnifiedForConditionalGeneration) can do both, so we load it once and route:

  * __call__  -> native tool-calling text generation (control tokens preserved,
                 stops at <tool_call|>/<turn|>) — the interface run_agent expects.
  * generate  -> clean text generation (e.g. the SVG markup the studio asks for).
  * ask_image -> vision: answer a prompt about a PIL image.

Text turns go through the tokenizer's chat template (proven to handle tools=);
image turns go through the full processor (which expands <|image|>).
"""

import os

import instrument
from instrument import METRICS, debug, note
from tokens import TOOL_CALL_CLOSE, TURN_CLOSE

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12b-it")


class UnifiedEngine:
    def __init__(self, model_id=MODEL_ID):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        note(f"loading {model_id} (unified vision+text engine)...")
        with instrument.Timer() as t:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="auto"
            )
        METRICS.model_load_s = t.elapsed
        note(f"model ready in {t.elapsed:.1f}s")

    def _gen(self, inputs, label, **kw):
        n_prompt = inputs["input_ids"].shape[1]
        debug(f"{label}: generating (prompt {n_prompt} tok)...")
        with instrument.Timer() as t:
            with self.torch.inference_mode():
                out = self.model.generate(**inputs, **kw)
        gen = out[0][n_prompt:]
        METRICS.record_generation(label, n_prompt, len(gen), t.elapsed)
        return gen

    # --- agent loop: native tool-calling text generation -----------------
    def __call__(self, messages, tools=None, enable_thinking=False):
        prompt = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self._gen(
            inputs, "agent", max_new_tokens=1024, temperature=0.2,
            tokenizer=self.tokenizer, stop_strings=[TOOL_CALL_CLOSE, TURN_CLOSE],
        )
        # Keep special tokens so <|tool_call> / <|"|> survive for the parser.
        return self.tokenizer.decode(gen, skip_special_tokens=False)

    # --- clean text generation (SVG markup, prose) -----------------------
    def generate(self, messages, max_new_tokens=2048):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        gen = self._gen(inputs, "text", max_new_tokens=max_new_tokens, do_sample=False)
        # skip_special_tokens removes the reserved control tokens but leaves
        # real markup like <svg>/<defs> intact (non-thinking mode emits no
        # channel text to clean).
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    # --- vision ----------------------------------------------------------
    def ask_image(self, image, prompt, max_new_tokens=320):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        gen = self._gen(inputs, "vision", max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(gen, skip_special_tokens=True).strip()
