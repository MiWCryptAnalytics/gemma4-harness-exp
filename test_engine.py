"""Engine-side logic that needs no GPU: KV-cache prefix matching, the stderr
streamer, and reasoning-channel extraction.

The cache decision is pure integer-list logic, so it is fully testable here —
probe_kv_cache.py covers the on-GPU behavior (identical output ± cache).
"""
import contextlib
import io
import sys

from engine import _common_prefix_len, _is_prefix_extension
from tokens import CHANNEL_CLOSE, CHANNEL_OPEN, THINK, clean, extract_channels

# --- KV cache reuse is allowed ONLY on an exact prefix extension -----------
base = [1, 2, 3, 4, 5]

# The normal agent step: the re-rendered prompt starts with everything the
# model has already seen, plus the new tool response.
assert _is_prefix_extension(base, base + [6, 7]), "extension must be reusable"

# The chat template re-serialized the assistant turn differently -> rebuild.
assert not _is_prefix_extension(base, [1, 2, 9, 4, 5, 6]), "divergence must miss"

# Degenerate cases: nothing cached, no new tokens, or a shorter prompt.
assert not _is_prefix_extension([], [1, 2])
assert not _is_prefix_extension(base, base), "equal length adds nothing to prefill"
assert not _is_prefix_extension(base, [1, 2, 3]), "shorter prompt can't reuse"
assert not _is_prefix_extension(base, []), "empty prompt can't reuse"
print("prefix-extension gate (hit / diverge / equal / shorter / empty): OK")

assert _common_prefix_len(base, [1, 2, 9]) == 2
assert _common_prefix_len(base, base) == 5
assert _common_prefix_len([], base) == 0
print("common-prefix length (debug attribution): OK")

# --- the streamer writes to stderr, never stdout --------------------------
# stdout carries the [Step N] lines the eval suite scrapes, and the eval driver
# grades stdout+stderr concatenated, so streaming is opt-in AND off stdout.
try:
    import transformers  # noqa: F401
    have_transformers = True
except ImportError:
    have_transformers = False

if have_transformers:
    from engine import _make_stderr_streamer

    class StubTokenizer:
        def decode(self, ids, **kw):
            return "".join(chr(96 + i) for i in ids)

    import torch

    out, err = io.StringIO(), io.StringIO()
    streamer = _make_stderr_streamer(StubTokenizer(), skip_prompt=False)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        streamer.put(torch.tensor([[1, 2, 3]]))
        streamer.put(torch.tensor([4]))
        streamer.end()
    assert out.getvalue() == "", f"streamer must not touch stdout: {out.getvalue()!r}"
    assert "abc" in err.getvalue(), err.getvalue()
    print("streamer -> stderr only:", repr(err.getvalue()))
else:
    print("transformers unavailable -> streamer test skipped")

# --- reasoning channel: extracted for the trace, still stripped for display -
reply = (f"{THINK}{CHANNEL_OPEN}thought\nThe user wants the disk size.\n"
         f"I should run df -h.{CHANNEL_CLOSE}The root filesystem has 20G free.")
chans = extract_channels(reply)
assert len(chans) == 1, chans
assert "run df -h" in chans[0] and "thought" not in chans[0].split("\n")[0], chans[0]
assert clean(reply) == "The root filesystem has 20G free.", clean(reply)
print("reasoning channel extracted:", repr(chans[0][:60]))
print("clean() still strips it:", repr(clean(reply)))

# Multiple channel blocks, and the no-thinking case.
multi = f"{CHANNEL_OPEN}thought\nfirst{CHANNEL_CLOSE}mid{CHANNEL_OPEN}thought\nsecond{CHANNEL_CLOSE}end"
assert extract_channels(multi) == ["first", "second"], extract_channels(multi)
assert extract_channels("no channels here") == []
assert extract_channels(f"{CHANNEL_OPEN}thought\n   {CHANNEL_CLOSE}done") == []
print("multi-block + empty-channel handling: OK")

print("\nall engine tests passed")
