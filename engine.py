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


def model_load_kwargs(quantize=None):
    """from_pretrained kwargs shared by every model loader in the harness.

    quantize=None (the default) loads full bf16. On a 24GB card the 12B's
    weights (~23GB) don't fit next to the KV cache, so device_map="auto"
    offloads some layers to CPU and generation is slow (~2 tok/s) — but output
    fidelity is maximal. That default is deliberate: at 4-bit the model often
    emits malformed SVG/XML, which breaks the structured-output tool workflows
    (svg_studio, create_image), so reduced precision is strictly opt-in:

      '8bit' — int8, ~13GB, fits a 24GB card fully on-GPU; mild fidelity loss.
      '4bit' — NF4, ~7GB, fastest (~30 tok/s on a 3090) but known to malform
               structured output.

    Quantized loads pin to one GPU so a model that doesn't fit OOMs loudly
    instead of silently offloading.
    """
    import torch
    if quantize is None:
        return dict(dtype=torch.bfloat16, device_map="auto")
    from transformers import BitsAndBytesConfig
    if quantize == "8bit":
        qc = BitsAndBytesConfig(load_in_8bit=True)
    elif quantize == "4bit":
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"quantize must be None, '4bit' or '8bit', got {quantize!r}")
    return dict(quantization_config=qc, dtype=torch.bfloat16, device_map="cuda:0")


class UnifiedEngine:
    def __init__(self, model_id=MODEL_ID, quantize=None):
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.torch = torch
        note(f"loading {model_id} (unified vision+text engine, "
             f"{quantize or 'bf16'})...")
        with instrument.Timer() as t:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id, **model_load_kwargs(quantize)
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
