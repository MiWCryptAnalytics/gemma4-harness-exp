"""Confirm the OUTPUT side: does Gemma 4 actually EMIT the reserved tool tokens?

probe_tokenizer.py proved the *input* side — the vocab has <|tool_call>, <|"|>,
etc., and the chat template renders tool *declarations* with them. What it did
NOT prove is that the model, when asked to use a tool, generates that native
grammar (rather than prose or our markdown convention).

This script settles it by running one real generation with the native tool
format and decoding WITHOUT stripping special tokens, so any reserved tokens the
model emits are visible. Raw token ids + decoded output are saved to
native_toolcall_probe.json for the record.

Needs the GPU. Run:  python probe_native_toolcall.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-12b-it"
OUTPUT = Path(__file__).resolve().parent / "native_toolcall_probe.json"

# Reserved tool-related token ids (from probe_tokenizer.py) we want to detect
# in the model's OUTPUT.
RESERVED = {
    48: "<|tool_call>", 49: "<tool_call|>",
    50: "<|tool_response>", 51: "<tool_response|>",
    46: "<|tool>", 47: "<tool|>", 52: '<|"|>',
}

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    messages = [{"role": "user", "content": "What's the weather in Tokyo right now?"}]
    prompt = tok.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()

    # Decode two ways: human-readable, and raw (special tokens visible).
    readable = tok.decode(gen_ids, skip_special_tokens=True)
    raw = tok.decode(gen_ids, skip_special_tokens=False)

    emitted_reserved = sorted({i for i in gen_ids if i in RESERVED})
    used = {i: RESERVED[i] for i in emitted_reserved}

    print("=== RAW generation (special tokens visible) ===")
    print(raw)
    print("\n=== reserved tool tokens the model EMITTED ===")
    print(used or "NONE — model did not use the native tool grammar")
    native = 48 in emitted_reserved  # <|tool_call> is the decisive marker
    print(f"\nVERDICT: model emits native tool-call grammar? -> {native}")

    # First ~40 generated tokens as (id, piece) for the record.
    head = [(i, tok.convert_ids_to_tokens(i)) for i in gen_ids[:40]]

    OUTPUT.write_text(json.dumps({
        "model": MODEL_ID,
        "prompt": prompt,
        "generated_text_readable": readable,
        "generated_text_raw": raw,
        "reserved_tokens_emitted": used,
        "emits_native_tool_call": native,
        "first_40_tokens": head,
    }, indent=2))
    print(f"\nSaved details to {OUTPUT}")


if __name__ == "__main__":
    main()
