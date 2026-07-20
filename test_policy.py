"""Unit tests for the web-policy engine (no Docker/GPU; needs PyYAML in venv)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sandbox", "squid"))

from policy import Req, load_policy  # noqa: E402


def _policy(text):
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(text)
        path = fh.name
    return load_policy(path)


def req(method="GET", scheme="https", host="example.com", path="/", headers=None):
    return Req(method=method, scheme=scheme, host=host, path=path, headers=headers or {})


# --- block by host glob and by url regex ---------------------------------
p = _policy("""
version: 1
default: allow
rules:
  - name: block-social
    match: { host: ["*.facebook.com", "x.com"] }
    action: block
    status: 403
    message: "nope"
  - name: block-secretpath
    match: { url: '^https?://[^/]+/admin/' }
    action: block
""")
d = p.decide_request(req(host="www.facebook.com"))
assert d.action == "block" and d.status == 403 and d.message == "nope", d
print("block by host glob: OK")

assert p.decide_request(req(host="x.com")).action == "block"
assert p.decide_request(req(host="example.com")).action == "allow", "unmatched host allowed"
assert p.decide_request(req(host="api.example.com", path="/admin/panel")).action == "block"
print("block by url regex + allow fallthrough: OK")

# --- redirect with $1 backref --------------------------------------------
p = _policy("""
version: 1
default: allow
rules:
  - name: redir
    match: { url: '^https?://old\\.example\\.com/(.*)$' }
    action: redirect
    location: 'https://new.example.com/$1'
""")
d = p.decide_request(req(host="old.example.com", path="/docs/page?x=1"))
assert d.action == "redirect", d
assert d.location == "https://new.example.com/docs/page?x=1", d.location
print("redirect with $1 backref: OK")

# --- set-header is non-terminal; accumulates, then allow -----------------
p = _policy("""
version: 1
default: allow
rules:
  - name: pin-ua
    match: { host: ["*"] }
    action: set-header
    request_headers:
      set: { User-Agent: "Gemma4-Agent/1.0" }
      remove: ["Cookie", "X-Telemetry"]
""")
d = p.decide_request(req(host="anything.test"))
assert d.action == "allow", d
assert d.header_ops["set"]["User-Agent"] == "Gemma4-Agent/1.0", d.header_ops
assert "Cookie" in d.header_ops["remove"] and "X-Telemetry" in d.header_ops["remove"]
print("set-header non-terminal accumulation: OK")

# --- default: deny blocks unmatched --------------------------------------
p = _policy("""
version: 1
default: deny
rules:
  - name: allow-good
    match: { host: ["good.example.com"] }
    action: allow
""")
assert p.decide_request(req(host="good.example.com")).action == "allow"
d = p.decide_request(req(host="evil.example.com"))
assert d.action == "block", d
print("default deny blocks unmatched: OK")

# --- rewrite: same-origin path rewrite + merge credential headers --------
p = _policy("""
version: 1
default: allow
rules:
  - name: offload
    match: { host: ["registry.npmjs.org"] }
    action: rewrite
    rewrite:
      path: { pattern: '^/(.*)$', replace: '/npm/$1' }
    request_headers:
      set: { Authorization: "Bearer TOK" }
""")
d = p.decide_request(req(host="registry.npmjs.org", path="/left-pad"))
assert d.action == "allow", d
assert d.rewrite is not None, d.rewrite
assert d.rewrite.apply_path("/left-pad") == "/npm/left-pad", d.rewrite.apply_path("/left-pad")
assert d.header_ops["set"]["Authorization"] == "Bearer TOK", d.header_ops
print("rewrite path+credentials: OK")

# rewrite is terminal but earlier set-header rules still accumulate
p = _policy("""
version: 1
default: allow
rules:
  - name: pin
    match: { host: ["*"] }
    action: set-header
    request_headers: { set: { User-Agent: "GM4" } }
  - name: rw
    match: { host: ["a.test"] }
    action: rewrite
    rewrite: { path: { pattern: '^/(.*)$', replace: '/x/$1' } }
""")
d = p.decide_request(req(host="a.test", path="/y"))
assert d.rewrite.apply_path("/y") == "/x/y" and d.header_ops["set"]["User-Agent"] == "GM4", d
print("rewrite merges earlier set-header ops: OK")

# --- rewrite-body: applies in-scope, no-op off-scope ---------------------
p = _policy("""
version: 1
default: allow
rules:
  - name: redact
    match: { host: ["*"], content_type: ["text/*", "application/json"] }
    action: rewrite-body
    response_body:
      - { pattern: '(?i)api_key\\s*=\\s*\\S+', replace: 'api_key=[REDACTED]' }
      - { pattern: '</body>', replace: '<x/></body>' }
""")
body = b"<html><body>api_key=SECRET123\n</body></html>"
out = p.decide_response(req(), {"content-type": "text/html; charset=utf-8"}, body)
assert out is not None, "expected rewrite"
assert b"[REDACTED]" in out and b"<x/></body>" in out and b"SECRET123" not in out, out
print("rewrite-body in-scope: OK")

# off-scope content-type: image/png -> untouched
out = p.decide_response(req(), {"content-type": "image/png"}, body)
assert out is None, "image body must not be rewritten"
print("rewrite-body off-scope no-op: OK")

# undecodable body under a text/* type -> left untouched (no corruption)
out = p.decide_response(req(), {"content-type": "text/html; charset=utf-8"}, b"\xff\xfe\xff")
assert out is None, "undecodable body must be left untouched"
print("rewrite-body skips undecodable bytes: OK")

# --- malformed policy raises (caller/reloader keeps last-good) ------------
try:
    _policy("default: [not, a, string]\nrules: 5\n")
    raised = False
except Exception:
    raised = True
assert raised, "malformed policy should raise from load_policy"
print("malformed policy raises: OK")

print("\nall policy engine tests passed")
