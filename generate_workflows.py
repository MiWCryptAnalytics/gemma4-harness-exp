"""Generate NATIVE Gemma workflows by recording the real agent loop.

Rather than hand-writing what we think Gemma's tool calls look like, we run the
actual model through the actual agent loop against the actual sandbox, and
record every turn it produces. The model's text is therefore in-distribution
for itself, and the tool outputs it reasons over are real (not hallucinated).

The captured turns are written to generated_workflows.json, which mockmodel.py
loads so `python gemma4.py --dry-run` replays genuine model output with no GPU.

Run this once on the GPU box:  python generate_workflows.py
"""

import json
from pathlib import Path

from sandbox import Sandbox
from gemma4 import build_transformers_engine, run_agent

OUTPUT = Path(__file__).resolve().parent / "generated_workflows.json"

# Seed tasks. The model decides the steps; we only pick the goals.
SEEDS = [
    {
        "name": "sysinfo",
        "system": (
            "You are a Linux administration assistant operating inside a "
            "sandboxed container. Inspect the environment with tools before "
            "answering. Be concise and avoid filler."
        ),
        "task": (
            "Identify the Linux distribution and kernel, then report disk space "
            "on the root filesystem."
        ),
    },
    {
        "name": "filecraft",
        "system": (
            "You are a coding assistant working in a sandboxed container. Use "
            "the tools to create and run code, then confirm the result."
        ),
        "task": (
            "Write a Python script that prints the first 10 Fibonacci numbers, "
            "run it, and confirm it works."
        ),
    },
    {
        "name": "datawrangle",
        "system": (
            "You are a data assistant in a sandboxed container. Build the data "
            "you need with the tools, then query it."
        ),
        "task": (
            "Create a JSON file of three users with ages, then report the name "
            "of the oldest user."
        ),
    },
]


class Recorder:
    """Wraps the engine to capture each reply the model generates."""

    def __init__(self, engine):
        self.engine = engine
        self.turns = []

    def __call__(self, messages, **kwargs):
        reply = self.engine(messages, **kwargs)
        self.turns.append(reply)
        return reply


def main():
    engine = build_transformers_engine()
    workflows = {}

    for seed in SEEDS:
        print(f"\n{'='*60}\n=== generating native workflow: {seed['name']}\n{'='*60}")
        recorder = Recorder(engine)
        # Fresh sandbox per workflow so recordings don't leak state into each other.
        with Sandbox() as _sb:
            run_agent(seed["system"], seed["task"], recorder)
        workflows[seed["name"]] = {
            "system": seed["system"],
            "task": seed["task"],
            "turns": recorder.turns,
            "source": "native (google/gemma-4-12b-it)",
        }

    OUTPUT.write_text(json.dumps(workflows, indent=2))
    print(f"\nWrote {len(workflows)} native workflows to {OUTPUT}")


if __name__ == "__main__":
    main()
