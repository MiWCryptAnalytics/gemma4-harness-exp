"""compose_music host output handling (no Docker/GPU needed).

The tool used to take a model-controlled `path` argument that was written on
the HOST and had to be contained to the CWD. That surface is gone: the schema
no longer advertises a path, a stray path argument is rejected gracefully, and
the WAV always lands at CWD/song.wav.
"""
import os
import tempfile
from pathlib import Path

import sandbox
from tools import REGISTRY, compose_music, dispatch

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

        # 1. The model-facing schema exposes only `abc` — no host path exists
        # for the model to traverse.
        params = REGISTRY["compose_music"]["schema"]["function"]["parameters"]
        assert set(params["properties"]) == {"abc"}, params
        assert params["required"] == ["abc"], params
        print("schema -> abc only")

        # 2. A stray `path` argument (e.g. hallucinated from an old prompt)
        # errors through dispatch before any sandbox or host write happens.
        sandbox._ACTIVE = None
        out = dispatch("compose_music", {"abc": ABC, "path": "/tmp/evil.wav"})
        assert out.startswith("Error calling compose_music"), out
        print("stray path argument -> rejected")

        # 3. Happy path through a stubbed sandbox lands at CWD/song.wav.
        stub = StubSandbox()
        sandbox._ACTIVE = stub
        try:
            out = compose_music(abc=ABC)
        finally:
            sandbox._ACTIVE = None
        assert (root / "song.wav").read_bytes() == b"RIFFfakewav", out
        assert "written to host song.wav" in out, out
        assert "sandbox copy at /workspace/_out.wav" in out, out
        assert any("_synth.py" in c for c in stub.commands), stub.commands
        print("compose_music happy path ->", out)
    finally:
        os.chdir(prev_cwd)

print("\nall compose_music output tests passed")
