"""Checks that the sandbox image trusts ONLY the MITM CA.

Needs Docker and the built sandbox image (`make sandbox-build`); skips cleanly
when either is absent so `make test` stays green on machines without them. This
is the cheap, offline half of the MITM verification — the end-to-end intercept
proof is `make mitm-verify` (needs the proxy + internet).
"""
import shutil
import subprocess
import sys

from sandbox import IMAGE_TAG


def _docker_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


def _image_built():
    return subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG], capture_output=True
    ).returncode == 0


if not _docker_available():
    print("docker unavailable: skipping CA-bundle test")
    sys.exit(0)
if not _image_built():
    print(f"{IMAGE_TAG} not built (run `make sandbox-build`): skipping CA-bundle test")
    sys.exit(0)

BUNDLE = "/etc/ssl/certs/ca-certificates.crt"

# 1. The system trust bundle must contain EXACTLY one certificate (our CA).
count = subprocess.run(
    ["docker", "run", "--rm", IMAGE_TAG, "sh", "-c",
     f"grep -c 'BEGIN CERTIFICATE' {BUNDLE}"],
    capture_output=True, text=True,
)
assert count.returncode == 0, count.stderr
n = count.stdout.strip()
assert n == "1", f"expected exactly 1 trusted CA, found {n}"
print("sandbox trusts exactly one CA: OK")

# 2. That one cert must be OUR MITM CA (subject carries the marker we set).
subj = subprocess.run(
    ["docker", "run", "--rm", IMAGE_TAG, "sh", "-c",
     f"openssl x509 -in {BUNDLE} -noout -subject"],
    capture_output=True, text=True,
)
assert subj.returncode == 0, subj.stderr
assert "Gemma4 Sandbox MITM CA" in subj.stdout, f"unexpected CA subject: {subj.stdout!r}"
print(f"the trusted CA is our MITM CA: OK  ({subj.stdout.strip()})")

# 3. The cert-trust env vars must be baked into the image so every tool uses them.
env = subprocess.run(
    ["docker", "run", "--rm", IMAGE_TAG, "sh", "-c",
     "echo $REQUESTS_CA_BUNDLE:$CURL_CA_BUNDLE:$SSL_CERT_FILE"],
    capture_output=True, text=True,
)
assert env.stdout.count(BUNDLE) == 3, f"cert env vars not all set: {env.stdout!r}"
print("curl/requests/ssl cert env vars point at the MITM bundle: OK")

print("\nall CA-bundle tests passed")
