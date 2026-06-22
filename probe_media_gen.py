"""Does Gemma 4 itself EMIT media (image/audio/video) tokens as output?

The vocab has open/close pairs for each modality (<|image>..<image|>,
<|audio>..<audio|>, <|video|>) plus a channel mechanism. Those might be purely
*input* placeholders, or the model might generate media through them. The only
way to know is to ask it and look at the raw token ids it produces.

This loads gemma 4 (text generation path) and, for each modality, prompts it to
generate that media, then reports any reserved media/channel tokens that appear
in the OUTPUT. No extra models — Gemma only.

Run:  python probe_media_gen.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-4-12b-it"
OUTPUT = Path(__file__).resolve().parent / "media_gen_probe.json"

# Reserved tokens that would indicate media OUTPUT (ids from probe_tokenizer.py).
MEDIA_TOKENS = {
    255999: "<|image>", 258880: "<|image|>", 258882: "<image|>",
    256000: "<|audio>", 258881: "<|audio|>", 258883: "<audio|>",
    258884: "<|video|>",
    98: "<|think|>", 100: "<|channel>", 101: "<channel|>",
}

PROMPTS = {
    "image": "Generate an image of a red circle on a white background.",
    "audio": "Generate a 3 second audio clip of a simple piano note.",
    "video": "Generate a short 2 second video of ocean waves.",
}


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )

    results = {}
    for kind, prompt in PROMPTS.items():
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()

        emitted = sorted({i for i in gen_ids if i in MEDIA_TOKENS})
        emitted_named = {i: MEDIA_TOKENS[i] for i in emitted}
        readable = tok.decode(gen_ids, skip_special_tokens=True)

        print(f"\n{'='*60}\n{kind.upper()}: {prompt}\n{'='*60}")
        print("media/channel tokens emitted:", emitted_named or "NONE")
        print("text reply (first 400 chars):")
        print(readable[:400])

        results[kind] = {
            "prompt": prompt,
            "media_tokens_emitted": emitted_named,
            "reply_text": readable,
        }

    any_media = any(
        any(i in MEDIA_TOKENS and i in (255999, 258880, 258882, 256000, 258881,
                                        258883, 258884)
            for i in [int(k) for k in r["media_tokens_emitted"]])
        for r in results.values()
    )
    print(f"\n=== VERDICT: Gemma emitted real media (not just channel) tokens? -> {any_media} ===")
    OUTPUT.write_text(json.dumps(results, indent=2))
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    main()
