"""Checks that Sandbox.export_workspace copies tmpfs artifacts to the host.

Needs Docker (it starts a real sandbox) but no GPU/model. Skips cleanly when
Docker is absent so `make test` stays green on machines without it.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sandbox import Sandbox


def _docker_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


if not _docker_available():
    print("docker unavailable: skipping export test")
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "exported"

    with Sandbox() as sb:
        # Write a text file and a nested binary-ish file inside the tmpfs workspace.
        sb.run("echo 'hello eval' > /workspace/a.txt")
        sb.run("mkdir -p /workspace/sub && printf '\\x89PNG' > /workspace/sub/img.png")
        sb.export_workspace(out)

    a = out / "a.txt"
    assert a.exists(), f"expected {a} to exist"
    assert a.read_text().strip() == "hello eval", repr(a.read_text())
    print("text artifact exported: OK")

    img = out / "sub" / "img.png"
    assert img.exists(), f"expected {img} to exist"
    assert img.read_bytes() == b"\x89PNG", repr(img.read_bytes())
    print("nested binary artifact exported: OK")

# An empty workspace must yield an empty (created) dir, not an error.
with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "empty"
    with Sandbox() as sb:
        sb.export_workspace(out)
    assert out.is_dir(), f"expected {out} to be created"
    assert not any(out.iterdir()), f"expected {out} to be empty, got {list(out.iterdir())}"
    print("empty workspace export: OK")

print("\nall export tests passed")
