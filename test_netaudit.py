"""Network-audit plumbing: the ICAP audit writer, body codecs, and the
host-side summary (no Docker needed — test_policy_e2e.py covers the live path).
"""
import gzip
import json
import os
import sys
import tempfile
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUDIT = os.path.join(tempfile.mkdtemp(), "audit.jsonl")
os.environ["GEMMA_AUDIT"] = _AUDIT  # read at import time

sys.path.insert(0, os.path.join(_HERE, "sandbox", "squid"))
import icap_server  # noqa: E402
from gemma4 import _summarize_net_audit  # noqa: E402

# --- audit(): one JSON line per call, appended -----------------------------
icap_server.audit(phase="reqmod", method="GET", host="example.org", path="/",
                  action="allow", rule="default")
icap_server.audit(phase="reqmod", method="GET", host="www.facebook.com", path="/",
                  action="block", rule="block-fb", status=403)
records = [json.loads(ln) for ln in open(_AUDIT) if ln.strip()]
assert len(records) == 2, records
assert records[1]["action"] == "block" and records[1]["rule"] == "block-fb"
assert records[0]["ts"] and records[1]["ts"], "every entry is timestamped"
print("audit writes one JSON line per decision: OK")

# --- audit() never raises, whatever happens -------------------------------
icap_server.AUDIT_PATH = "/nonexistent-dir/audit.jsonl"
icap_server.audit(phase="reqmod", host="x")     # must not raise
icap_server.AUDIT_PATH = _AUDIT
print("audit failure is swallowed (can't take the proxy down): OK")

# --- the size cap stops writing rather than filling the disk --------------
icap_server.AUDIT_MAX_BYTES = 1
icap_server._audit_state.update(bytes=0, capped=False)
icap_server.audit(phase="reqmod", host="capped.test")
assert icap_server._audit_state["capped"], "oversized write should trip the cap"
assert "capped.test" not in open(_AUDIT).read()
icap_server.AUDIT_MAX_BYTES = 50 * 1024 * 1024
icap_server._audit_state.update(bytes=0, capped=False)
print("audit size cap: OK")

# --- decode_body: stdlib codecs round-trip; unknown ones pass through -----
payload = b"<html><body>hello</body></html>"
for enc, comp in (("", lambda b: b), ("identity", lambda b: b),
                  ("gzip", gzip.compress), ("deflate", zlib.compress)):
    decoded, recompress = icap_server.decode_body(comp(payload), enc)
    assert decoded == payload, enc
    again, _ = icap_server.decode_body(recompress(payload), enc or "identity")
    assert again == payload, f"{enc} round-trip"
print("decode_body gzip/deflate/identity round-trip: OK")

for enc, mod in (("zstd", icap_server.zstandard), ("br", icap_server.brotli)):
    if mod is None:
        # Absent codec must degrade to pass-through, never raise.
        assert icap_server.decode_body(b"\x00garbage", enc) == (None, None), enc
        print(f"decode_body {enc}: not installed here -> pass-through (OK; the "
              "proxy image installs python3-zstandard/python3-brotli)")
        continue
    comp = (mod.ZstdCompressor().compress if enc == "zstd" else mod.compress)
    decoded, recompress = icap_server.decode_body(comp(payload), enc)
    assert decoded == payload, enc
    again, _ = icap_server.decode_body(recompress(payload), enc)
    assert again == payload, f"{enc} round-trip"
    print(f"decode_body {enc} round-trip: OK")

assert icap_server.decode_body(b"whatever", "exotic-codec") == (None, None)
print("decode_body unknown encoding -> pass-through: OK")

# --- host-side summary over a realistic audit log -------------------------
log = "\n".join(json.dumps(r) for r in [
    {"phase": "reqmod", "host": "example.org", "action": "allow"},
    {"phase": "reqmod", "host": "example.org", "action": "allow"},
    {"phase": "reqmod", "host": "www.facebook.com", "action": "block"},
    {"phase": "reqmod", "host": "old.test", "action": "redirect"},
    {"phase": "respmod", "host": "example.org", "action": "rewrite-body"},
    {"phase": "respmod", "host": "example.org", "action": "pass"},
]) + "\n"
s = _summarize_net_audit(log)
assert s == {"requests": 4, "blocked": 1, "redirected": 1,
             "bodies_rewritten": 1, "hosts": 3}, s
assert _summarize_net_audit("not json\n\n")["requests"] == 0, "tolerate junk lines"
print("net audit summary:", s)

print("\nall network-audit tests passed")
