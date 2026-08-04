"""Host output-path containment for tools that copy artifacts out of the
sandbox (no Docker/GPU needed).

compose_music's `path` argument is model-controlled and is written on the HOST,
so it must be confined to the working directory: traversal, absolute paths, and
symlink escapes are rejected before any sandbox work happens.
"""
import os
import tempfile
from pathlib import Path

import sandbox
import tools
from tools import _host_output_path, compose_music

ABC = "K:C\nA"


class StubSandbox:
    """Just enough of sandbox.Sandbox for compose_music's happy path."""

    def __init__(self):
        self.commands = []

    def run(self, command, stdin=None):
        self.commands.append(command)
        return sandbox.ToolResult(0, "synthesized 1 notes")

    def read_bytes(self, container_path):
        return b"RIFFfakewav"


with tempfile.TemporaryDirectory() as tmp:
    prev_cwd = os.getcwd()
    os.chdir(tmp)
    try:
        root = Path(tmp).resolve()

        # 1. _host_output_path: legitimate relative paths resolve inside CWD.
        assert _host_output_path("song.wav") == root / "song.wav"
        assert _host_output_path("out/tunes/song.wav") == root / "out" / "tunes" / "song.wav"
        print("relative paths -> contained in", root)

        # 2. Absolute and traversal paths are rejected.
        for bad in ("/tmp/evil.wav", "../evil.wav", "a/../../evil.wav",
                    "..", "../../etc/passwd"):
            assert _host_output_path(bad) is None, bad
        print("absolute + '..' paths -> rejected")

        # 3. A symlink pointing outside the CWD can't be used to escape.
        outside = Path(tempfile.mkdtemp())
        (root / "link").symlink_to(outside)
        assert _host_output_path("link/evil.wav") is None
        print("symlink escape -> rejected")

        # 4. compose_music refuses bad paths BEFORE touching the sandbox:
        # with no active sandbox this must return an error, not raise.
        sandbox._ACTIVE = None
        for bad in ("../evil.wav", "/tmp/evil.wav"):
            out = compose_music(abc=ABC, path=bad)
            assert isinstance(out, str) and out.startswith("Error:"), out
        assert not (Path(tmp).parent / "evil.wav").exists()
        print("compose_music traversal -> refused with no sandbox call")

        # 5. Happy path through a stubbed sandbox lands inside the CWD.
        stub = StubSandbox()
        sandbox._ACTIVE = stub
        try:
            out = compose_music(abc=ABC, path="out/song.wav")
        finally:
            sandbox._ACTIVE = None
        dest = root / "out" / "song.wav"
        assert dest.read_bytes() == b"RIFFfakewav", out
        assert "sandbox copy at /workspace/_out.wav" in out, out
        assert any("_synth.py" in c for c in stub.commands), stub.commands
        print("compose_music happy path ->", out)
    finally:
        os.chdir(prev_cwd)

print("\nall tool path-containment tests passed")
