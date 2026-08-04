"""End-to-end browse() through the real sandbox + MITM proxy.

Needs Docker, the built images (`make sandbox-build`), the MITM CA, and internet.
Skips cleanly when any is absent, mirroring test_policy_e2e.py's gating.

Proves the point of routing the fetch through the sandbox rather than the host:
  - a real page comes back as readable text
  - the web policy governs browse() with no special-casing (a blocked host
    returns the operator's block message, not a network error)
  - the request appears in the proxy's network audit
"""
import json
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
    print("docker unavailable: skipping browse e2e test")
    sys.exit(0)
if not (_image(IMAGE_TAG) and _image(PROXY_IMAGE_TAG)):
    print(f"{IMAGE_TAG}/{PROXY_IMAGE_TAG} not built (run `make sandbox-build`): "
          "skipping browse e2e test")
    sys.exit(0)
if not os.path.isfile(_CA):
    print("MITM CA missing (run `make mitm-ca`): skipping browse e2e test")
    sys.exit(0)

POLICY = """\
version: 1
default: allow
rules:
  - name: block-social
    match: { host: ["*.facebook.com"] }
    action: block
    status: 403
    message: "BROWSE-E2E-BLOCKED"
"""

failures = []


def check(name, ok, detail=""):
    print(f"{name}: {'OK' if ok else 'FAIL'}  {detail}")
    if not ok:
        failures.append(name)


with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
    fh.write(POLICY)
    policy_path = fh.name
os.chmod(policy_path, 0o644)

with Sandbox(network=True, exec_timeout=60, policy_file=policy_path) as sb:
    import web_tool  # registers browse; needs an active sandbox to run
    from tools import REGISTRY

    assert "browse" in REGISTRY, "importing web_tool must register browse"

    # 1. a real page renders as readable text (not raw HTML)
    out = str(web_tool.browse("https://example.org/"))
    check("browse returns readable text",
          "Example Domain" in out and "<html" not in out.lower(),
          f"({out.splitlines()[0] if out else '<empty>'})")
    check("browse reports the status line", "Status: 200" in out,
          f"({[l for l in out.splitlines() if l.startswith('Status')]})")

    # 2. the policy governs browse with no special-casing: the agent sees the
    #    operator's block page, which is the useful signal.
    out = str(web_tool.browse("https://www.facebook.com/"))
    check("policy block reaches browse", "BROWSE-E2E-BLOCKED" in out,
          f"({out[:120]!r})")
    check("blocked fetch reports 403", "Status: 403" in out, f"({out[:80]!r})")

    # 3. a question focuses the excerpt
    out = str(web_tool.browse("https://example.org/", "what is this domain for?"))
    check("question-focused fetch", "Status: 200" in out and len(out) > 40,
          f"({len(out)} chars)")

    # 4. every browse shows up in the proxy's network audit
    audit = sb.collect_net_audit()
    if not audit:
        check("browse appears in the audit", False, "(no audit collected)")
    else:
        records = [json.loads(ln) for ln in audit.splitlines() if ln.strip()]
        hosts = {r.get("host") for r in records}
        check("browse traffic audited", "example.org" in hosts, f"(hosts={sorted(hosts)})")
        blocked = [r for r in records
                   if r.get("action") == "block" and "facebook" in (r.get("host") or "")]
        check("blocked browse audited", bool(blocked),
              f"(rule={blocked[0]['rule']!r})" if blocked else "")

os.unlink(policy_path)

if failures:
    print(f"\nbrowse e2e FAILED: {failures}")
    sys.exit(1)
print("\nall browse e2e checks passed")
