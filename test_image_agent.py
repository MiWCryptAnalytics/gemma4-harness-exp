"""No-GPU wiring test: agent -> create_image -> render -> score -> present.

A FakeEngine stands in for Gemma (text tool-call, SVG text, vision score), but
the SVG rendering is REAL (rsvg-convert), so this exercises the full integration
path except the model itself.
"""


class FakeEngine:
    """Mimics UnifiedEngine's three methods with canned, deterministic output."""

    def __call__(self, messages, tools=None, enable_thinking=False):
        # Agent turn: first call -> emit a native tool call; then -> final answer.
        idx = sum(1 for m in messages if m["role"] == "assistant")
        if idx == 0:
            return ('<|tool_call>call:create_image{'
                    'description:<|"|>a friendly orange cat<|"|>}<tool_call|>')
        return "Here is your image (scored 8/10): see the saved image_path. Enjoy!"

    def generate(self, messages, max_new_tokens=2048):
        text = " ".join(m["content"] for m in messages if isinstance(m["content"], str))
        if "yes/no" in text.lower():  # checklist request
            return "Is there a cat?\nIs it orange?\nIs the background blue?"
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
                '<rect width="100" height="100" fill="skyblue"/>'
                '<circle cx="50" cy="55" r="28" fill="orange"/>'
                '<circle cx="42" cy="50" r="3" fill="black"/>'
                '<circle cx="58" cy="50" r="3" fill="black"/></svg>')

    def ask_image(self, image, prompt, max_new_tokens=320):
        return "yes"  # perceptual checks all pass


def main():
    import image_tool  # registers create_image
    from gemma4 import run_agent
    from sandbox import Sandbox
    from tools import REGISTRY

    fake = FakeEngine()
    image_tool.set_engine(fake)
    schema = [REGISTRY["create_image"]["schema"]]

    with Sandbox():  # SVG is now rasterized inside the sandbox
        answer = run_agent(
            "You make images and only show ones scoring >= 7.",
            "Please draw me a friendly cat.",
            fake, tools=schema,
        )
    assert answer, "agent returned no final answer"
    assert image_tool.LAST_IMAGE and image_tool.LAST_IMAGE["path"], "no image delivered"
    print("\n[test] PASS — sandboxed render + score + present; "
          f"delivered {image_tool.LAST_IMAGE['path']}")


if __name__ == "__main__":
    main()
