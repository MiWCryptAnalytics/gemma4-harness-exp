"""End-to-end web-policy test through the real MITM proxy + ICAP server.

Needs Docker, the built images (`make sandbox-build`), the MITM CA, and internet.
Skips cleanly when any is absent so `make test` stays green without them. Mirrors
test_ca_bundle.py's gating.

Proves, via a temp policy driven into the proxy with Sandbox(policy_file=...):
  - block   : a blocked host returns the synthesized 403 page (no origin contact)
  - redirect: a matched host 302s to the rewritten Location (no origin contact)
  - respmod : a fetched HTML body has the injected marker (real origin + rewrite)
  - sethdr  : the pinned User-Agent reaches an echo origin (soft; needs httpbin)
"""
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandbox import IMAGE_TAG, PROXY_IMAGE_TAG, Sandbox  # noqa: E402


def _docker_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _image(tag):
    return subprocess.run(["docker", "image", "inspect", tag],
                          capture_output=True).returncode == 0


_HERE = os.path.dirname(os.path.abspath(__file__))
_CA = os.path.join(_HERE, "sandbox", "mitm", "ca.crt")

if not _docker_available():
    print("docker unavailable: skipping policy e2e test")
    sys.exit(0)
if not (_image(IMAGE_TAG) and _image(PROXY_IMAGE_TAG)):
    print(f"{IMAGE_TAG}/{PROXY_IMAGE_TAG} not built (run `make sandbox-build`): "
          "skipping policy e2e test")
    sys.exit(0)
if not os.path.isfile(_CA):
    print("MITM CA missing (run `make mitm-ca`): skipping policy e2e test")
    sys.exit(0)

POLICY = """\
version: 1
default: allow
rules:
  - name: block-fb
    match: { host: ["*.facebook.com"] }
    action: block
    status: 403
    message: "E2E-BLOCKED-MARKER"
  - name: redirect-ex
    match: { url: '^https?://example\\.com/(.*)$' }
    action: redirect
    location: 'https://example.org/$1'
  - name: pin-ua
    match: { host: ["httpbin.org"] }
    action: set-header
    request_headers: { set: { User-Agent: "GM4-E2E-UA" } }
  - name: inject
    match: { host: ["example.org"], content_type: ["text/*"] }
    action: rewrite-body
    response_body:
      - { pattern: '</body>', replace: '<!--GM4-INJECTED--></body>' }
  - name: rewrite-path-and-creds
    match: { host: ["httpbin.org"], url: '/status/' }
    action: rewrite
    rewrite:
      path: { pattern: '^/status/[0-9]+$', replace: '/get' }
    request_headers:
      set: { Authorization: "Bearer OFFLOAD-TOK" }
"""

failures = []


def check(name, ok, detail=""):
    if ok:
        print(f"{name}: OK  {detail}")
    else:
        print(f"{name}: FAIL  {detail}")
        failures.append(name)


with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
    fh.write(POLICY)
    policy_path = fh.name
os.chmod(policy_path, 0o644)  # ICAP server runs as the squid uid; must be readable

with Sandbox(network=True, exec_timeout=40, policy_file=policy_path) as sb:
    # 1. block — REQMOD returns a 403 page; origin never contacted.
    r = sb.run("curl -s -o /dev/null -w '%{http_code}' https://www.facebook.com/")
    check("block status 403", r.output.strip() == "403", f"(got {r.output.strip()})")
    r = sb.run("curl -s https://www.facebook.com/ | grep -o E2E-BLOCKED-MARKER | head -1")
    check("block page body", "E2E-BLOCKED-MARKER" in r.output, f"(got {r.output.strip()!r})")

    # 2. redirect — REQMOD returns a 302 to the rewritten Location.
    r = sb.run("curl -s -o /dev/null -w '%{http_code}|%{redirect_url}' https://example.com/foo")
    out = r.output.strip()
    check("redirect 302 + location", out == "302|https://example.org/foo", f"(got {out!r})")

    # 3. respmod — real fetch of example.org, body gets the injected marker.
    r = sb.run("curl -s https://example.org/ | grep -o GM4-INJECTED | head -1")
    check("respmod body injection", "GM4-INJECTED" in r.output, f"(got {r.output.strip()!r})")

    # 4. set-header — pinned UA reaches an echo origin (soft; httpbin can be down).
    r = sb.run("curl -s -m15 https://httpbin.org/headers")
    if not r.output.strip() or r.exit_code != 0:
        print("setheader pin: SKIP (httpbin unreachable)")
    else:
        pinned = "GM4-E2E-UA" in r.output
        check("setheader pin", pinned,
              "(UA echoed)" if pinned else f"(got {r.output.strip()[:120]!r})")

    # 5. in-place rewrite — same-origin path rewrite + credential injection.
    #    /status/418 (normally 418) is rewritten to /get, and Authorization is
    #    injected; httpbin echoes both back. (Cross-origin host reroute is
    #    pinned by TLS-intercept; see README.)
    r = sb.run("curl -s -m15 https://httpbin.org/status/418")
    if not r.output.strip():
        print("rewrite path+creds: SKIP (httpbin unreachable)")
    else:
        rewritten = '"url"' in r.output and "httpbin.org/get" in r.output
        check("rewrite same-origin path", rewritten, "(/status/418 -> /get)")
        check("rewrite credential inject", "OFFLOAD-TOK" in r.output, "(Authorization echoed)")

os.unlink(policy_path)

if failures:
    print(f"\npolicy e2e FAILED: {failures}")
    sys.exit(1)
print("\nall policy e2e checks passed")
