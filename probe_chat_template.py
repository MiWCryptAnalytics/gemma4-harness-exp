"""Dump Gemma 4's chat template + render every scenario the native harness needs.

Before rewriting the harness around reserved tokens, we need the template's
exact serialization for: tool declarations, an assistant turn that CALLS a tool,
a tool RESPONSE turn, and multimodal (image/audio/video) content. The Jinja
template is the ground truth; this prints it and renders concrete examples so we
build/parse the precise byte format the model was trained on.

Run:  python probe_chat_template.py
"""

from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "forecast days"},
            },
            "required": ["location"],
        },
    },
}]


def banner(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


def render(tok, messages, **kw):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True, **kw)
    except Exception as exc:
        return f"(render failed: {type(exc).__name__}: {exc})"


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    banner("RAW chat_template (Jinja source)")
    print(tok.chat_template)

    banner("1) tool declaration + user turn")
    print(render(tok, [{"role": "user", "content": "Weather in Tokyo?"}], tools=TOOLS))

    banner("2) multi-turn: assistant CALLS tool, then tool RESPONSE")
    convo = [
        {"role": "user", "content": "Weather in Tokyo?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {
                "name": "get_weather", "arguments": {"location": "Tokyo", "days": 3}}}
        ]},
        {"role": "tool", "name": "get_weather", "content": '{"temp_c": 22, "sky": "clear"}'},
    ]
    print(render(tok, convo, tools=TOOLS))

    banner("3) multimodal user content (image + text)")
    mm = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What is in this image?"},
    ]}]
    print(render(tok, mm))

    banner("4) multimodal user content (audio + text)")
    au = [{"role": "user", "content": [
        {"type": "audio"},
        {"type": "text", "text": "Transcribe this."},
    ]}]
    print(render(tok, au))


if __name__ == "__main__":
    main()
