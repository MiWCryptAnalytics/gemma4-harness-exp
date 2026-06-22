"""An agent that creates images for the user — and checks quality before showing.

Wires the native tool-calling loop to a single tool, create_image, backed by a
shared UnifiedEngine. The model receives the user's request, calls create_image
(which runs the SVG self-critique loop and returns a quality score), then decides
whether the result is good enough to present.

    python image_agent.py --request "Show me a picture of a lighthouse at night."
    python image_agent.py --mock      # wiring smoke test, no GPU
"""

import argparse

import image_tool  # registers create_image
from gemma4 import run_agent
from tools import REGISTRY

SYSTEM = (
    "You are a helpful assistant that can create pictures with the create_image "
    "tool. When the user asks for an image, call create_image with a vivid, "
    "detailed description. The tool returns a quality_score out of 10. Only "
    "present the image (state its image_path) if quality_score is at least 7; if "
    "it is lower, apologize and say you could not make a good enough image."
)


def _image_tool_schema():
    return [REGISTRY["create_image"]["schema"]]


def main():
    ap = argparse.ArgumentParser(description="Gemma image-making agent")
    ap.add_argument("--request", default="Show me a picture of a lighthouse on a cliff at night.")
    ap.add_argument("--mock", action="store_true",
                    help="Run the wiring with a fake engine (no GPU).")
    args = ap.parse_args()

    if args.mock:
        from test_image_agent import FakeEngine
        engine = FakeEngine()
    else:
        from engine import UnifiedEngine
        engine = UnifiedEngine()

    image_tool.set_engine(engine)
    # The sandbox is needed for the in-container SVG rasterization.
    from sandbox import Sandbox
    with Sandbox():
        run_agent(SYSTEM, args.request, engine, tools=_image_tool_schema())

    # Deliver the result to the user, independent of the agent's phrasing.
    if image_tool.LAST_IMAGE and image_tool.LAST_IMAGE["path"]:
        r = image_tool.LAST_IMAGE
        print("\n" + "=" * 60)
        print(f"IMAGE DELIVERED TO USER: {r['path']}")
        print(f"  self-assessed quality: {r['score']}/10")
        print("=" * 60)
    else:
        print("\n[no image was delivered]")


if __name__ == "__main__":
    main()
