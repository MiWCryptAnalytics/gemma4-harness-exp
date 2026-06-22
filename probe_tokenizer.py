"""Empirically probe how Gemma 4 IT represents tool calls at the token level.

Motivating question: when Gemma emits a tool call, is the framing made of
dedicated *reserved control tokens* (like <|tool_call>, <|"|>), or is the whole
thing ordinary text that a parser scrapes afterward?

We don't have to argue from memory — the tokenizer is the ground truth. This
script inspects the actual added/special vocabulary and checks whether the
claimed tool tokens are single reserved IDs or just split into many text
tokens. Loads only the tokenizer (fast, no GPU).

Run:  python probe_tokenizer.py
"""

from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-12b-it"

# Tokens a popular explanation claimed Gemma 4 uses to frame tool calls.
CLAIMED_TOOL_TOKENS = ["<|tool_call>", "<tool_call|>", "<|tool_response>", '<|"|>']
# Tokens/markers we actually rely on in this harness, for comparison.
KNOWN_TOKENS = ["<start_of_turn>", "<end_of_turn>", "```tool_code", "```tool_output"]


def classify(tok, s):
    """Return (is_single_reserved_token, token_pieces) for a string."""
    ids = tok.encode(s, add_special_tokens=False)
    return len(ids) == 1, tok.convert_ids_to_tokens(ids)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"=== {MODEL_ID} — vocab size {len(tok)} ===\n")

    # 1) Scan every added/special token for anything tool-related.
    added = tok.get_added_vocab()  # str -> id
    toolish = {
        t: i for t, i in added.items()
        if any(k in t.lower() for k in ("tool", "call", "response", "func"))
    }
    print(f"# added/special tokens total: {len(added)}")
    print(f"tool-related reserved tokens: {toolish or 'NONE FOUND'}\n")

    # 2) Are the *claimed* framing tokens real single reserved tokens?
    print("Claimed tool-framing tokens — single reserved token, or plain text?")
    for s in CLAIMED_TOOL_TOKENS:
        single, pieces = classify(tok, s)
        verdict = "SINGLE reserved token" if single else f"{len(pieces)} text tokens (NOT reserved)"
        print(f"  {s!r:20} -> {verdict}: {pieces}")
    print()

    # 3) For contrast: tokens this harness actually uses.
    print("Tokens/markers this harness uses:")
    for s in KNOWN_TOKENS:
        single, pieces = classify(tok, s)
        kind = "reserved" if single else f"{len(pieces)} text tokens"
        print(f"  {s!r:18} -> {kind}: {pieces}")

    # 4) Full dump of every reserved/added token, ordered by id.
    print("\nAll added/reserved tokens (id: token):")
    for t, i in sorted(added.items(), key=lambda kv: kv[1]):
        print(f"  {i:>6}: {t!r}")

    # 5) Resolve the start_of_turn puzzle: is it a reserved token at all here?
    print("\nIs '<start_of_turn>' a known token id?",
          tok.convert_tokens_to_ids("<start_of_turn>"))
    print("(unk id is", tok.unk_token_id, ")")

    # 6) Render the chat template WITH tools to reveal the native tool format
    #    the model was actually trained to emit.
    tools = [{
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    }]
    messages = [{"role": "user", "content": "Create hello.py that prints hi."}]
    print("\n=== chat template rendered WITH tools ===")
    try:
        rendered = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        print(rendered)
    except Exception as exc:
        print(f"(template does not accept tools=: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
