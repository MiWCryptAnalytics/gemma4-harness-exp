"""No-GPU plumbing test for look_at: sandbox image -> bytes -> PIL -> engine."""
from sandbox import Sandbox
from svg_studio import render_svg
import vision_tool


class FakeVision:
    def ask_image(self, image, question, max_new_tokens=320):
        assert image.size == (128, 128), image.size  # bytes really decoded
        return f"[fake-vision {image.size}] you asked: {question}"


def main():
    vision_tool.set_engine(FakeVision())
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
           '<circle cx="50" cy="50" r="40" fill="red"/></svg>')
    with Sandbox() as s:
        # Leaves /workspace/_render.png inside the container.
        render_svg(svg, "host_copy.png", size=128)
        out = vision_tool.look_at("_render.png", "What shape is this?")
        print("look_at ->", out)
        assert "fake-vision" in out and "(128, 128)" in out

        missing = vision_tool.look_at("does_not_exist.png")
        print("missing ->", missing)
        assert missing.startswith("Error")
    print("PASS — look_at read the sandbox image, decoded it, reached the engine")


if __name__ == "__main__":
    main()
