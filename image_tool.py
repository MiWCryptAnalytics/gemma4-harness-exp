"""The create_image tool: agent-callable, quality-gated image synthesis.

Registering this (just importing the module) adds `create_image` to the tool
registry, so the native tool-calling agent can invoke it like any other tool.
Internally it runs the SVGStudio generate->render->self-critique->refine loop and
returns the BEST attempt together with Gemma's own quality score — so the agent
can decide whether the result is good enough to show the user.

The studio shares the agent's model via set_engine(), so no extra VRAM.
"""

from tools import tool
from svg_studio import SVGStudio

_STUDIO = None

# The most recent image the tool produced, so the harness can deliver it to the
# user regardless of how the agent phrases its final answer.
LAST_IMAGE = None


def set_engine(engine):
    """Point the image tool at a (shared) engine before running the agent."""
    global _STUDIO
    _STUDIO = SVGStudio(engine=engine)


@tool
def create_image(description: str, min_score: int = 7):
    """Create an image from a text description, self-assessing its quality (1-10)."""
    global LAST_IMAGE
    if _STUDIO is None:
        return "Error: image engine not initialized (call set_engine first)."
    r = _STUDIO.create(description, min_score=min_score)
    if not r["path"]:
        return f"Failed to produce a valid image for: {description!r}."
    LAST_IMAGE = r
    return (
        f"image_path={r['path']} quality_score={r['score']}/10 "
        f"attempts={r['attempts']} assessment={r['assessment']!r}"
    )
