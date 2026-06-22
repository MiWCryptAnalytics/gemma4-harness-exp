"""look_at — gives the agent EYES.

The agent's hands (shell/file tools) act in the sandbox; this tool lets it SEE
files there too. It reads an image out of the sandbox (sandbox.read_bytes, which
works with our tmpfs workspace) and runs it through Gemma 4's own vision tower
via the shared UnifiedEngine — so the same model that drives the agent loop also
looks at the picture. Uniquely possible because gemma-4 is a unified model.

Registering it (importing this module) adds `look_at` to the tool registry. Call
set_engine() with the agent's UnifiedEngine first so the tool can see.
"""

from io import BytesIO

from sandbox import get_active
from tools import tool

_ENGINE = None


def set_engine(engine):
    """Point the vision tool at the agent's (vision-capable) engine."""
    global _ENGINE
    _ENGINE = engine


@tool
def look_at(path: str, question: str = "Describe this image in detail."):
    """Visually inspect an image file in the sandbox and answer a question about it."""
    if _ENGINE is None:
        return "Error: vision engine not initialized."
    container_path = path if path.startswith("/") else f"/workspace/{path}"
    try:
        data = get_active().read_bytes(container_path)
    except RuntimeError as exc:
        return f"Error: could not read {path} ({exc})."
    try:
        from PIL import Image
        img = Image.open(BytesIO(data))
        img.load()
    except Exception as exc:
        return f"Error: {path} is not a readable image ({type(exc).__name__}: {exc})."
    return _ENGINE.ask_image(img, question)
