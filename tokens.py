"""Gemma 4 reserved control tokens, in one place.

These are real single-id tokens in the model's vocabulary (verified by
probe_tokenizer.py). The whole harness is built around emitting/parsing these
exact strings rather than an ad-hoc markdown convention — the model was trained
on them, so tool calls round-trip without the escaping/quoting ambiguity that
plain-text formats suffer from.
"""

# Turn framing.  <|turn>ROLE\n ... <turn|>\n   (assistant role is rendered "model")
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"

# Tool declarations (system block):  <|tool>declaration:NAME{...}<tool|>
TOOL_DECL_OPEN = "<|tool>"
TOOL_DECL_CLOSE = "<tool|>"

# A tool call the model emits:  <|tool_call>call:NAME{k:v,...}<tool_call|>
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"

# A tool result fed back:  <|tool_response>response:NAME{...}<tool_response|>
TOOL_RESPONSE_OPEN = "<|tool_response>"
TOOL_RESPONSE_CLOSE = "<tool_response|>"

# String-value delimiter.  A reserved token, so it can't collide with content —
# this is exactly why native parsing is robust where our old text parser wasn't.
STR_DELIM = '<|"|>'

# Reasoning channel.  enable_thinking opens <|think|>; the model brackets its
# private reasoning as <|channel>thought\n ... <channel|>.
THINK = "<|think|>"
CHANNEL_OPEN = "<|channel>"
CHANNEL_CLOSE = "<channel|>"

# Multimodal placeholders (the processor expands these into media embeddings).
IMAGE = "<|image|>"
AUDIO = "<|audio|>"
VIDEO = "<|video|>"


import re as _re

# The set of control tokens to strip from model output for display. Listing them
# explicitly (rather than a broad <...> regex) is deliberate: a greedy pattern
# would also eat real markup like <svg>/<defs> in generated SVG.
_KNOWN_TOKENS = [
    THINK, CHANNEL_OPEN, CHANNEL_CLOSE, TURN_OPEN, TURN_CLOSE,
    TOOL_CALL_OPEN, TOOL_CALL_CLOSE, TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE,
    TOOL_DECL_OPEN, TOOL_DECL_CLOSE, STR_DELIM, IMAGE, AUDIO, VIDEO,
]
_CHANNEL_RE = _re.compile(_re.escape(CHANNEL_OPEN) + r".*?" + _re.escape(CHANNEL_CLOSE), _re.DOTALL)


def clean(text):
    """Strip the reasoning channel + known control tokens, leaving prose/markup."""
    text = _CHANNEL_RE.sub("", text)
    for tok in _KNOWN_TOKENS:
        text = text.replace(tok, "")
    return text.strip()
