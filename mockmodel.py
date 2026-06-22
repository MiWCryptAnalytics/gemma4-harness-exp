"""A scripted stand-in for the Gemma model, for --dry-run.

Same interface as the real engine: a callable mapping the conversation to the
model's next raw reply. Each scripted turn is native Gemma output — either a
`<|tool_call>call:NAME{...}<tool_call|>` block (to act) or a plain final answer.

The SYNTHESIZED workflows below are a no-GPU fallback. If generate_workflows.py
has been run, the NATIVE recordings in generated_workflows.json are loaded and
take precedence — that's the in-distribution data we actually want.
"""

import json
from pathlib import Path

_NATIVE_FILE = Path(__file__).resolve().parent / "generated_workflows.json"


class MockModel:
    """Replays a scripted sequence of assistant turns."""

    def __init__(self, turns):
        self.turns = turns

    def __call__(self, messages, **kwargs):
        # One assistant turn is appended per tool-call round, so the count of
        # assistant turns already present selects the next scripted reply.
        idx = sum(1 for m in messages if m["role"] == "assistant")
        if idx < len(self.turns):
            return self.turns[idx]
        return "Workflow complete."  # graceful end if the loop outruns the script


# Native-grammar fallback workflows.  Strings are wrapped in the <|"|> delimiter.
SYNTHESIZED = {
    "sysinfo": {
        "system": (
            "You are a Linux administration assistant in a sandboxed container. "
            "Inspect the environment with tools before answering."
        ),
        "task": "Identify the distro and kernel, then report root-fs disk space.",
        "turns": [
            '<|tool_call>call:shell{command:<|"|>cat /etc/os-release<|"|>}<tool_call|>',
            '<|tool_call>call:shell{command:<|"|>uname -srm<|"|>}<tool_call|>',
            '<|tool_call>call:shell{command:<|"|>df -h /<|"|>}<tool_call|>',
            "This container runs Debian GNU/Linux 13 (trixie); see the df output "
            "for the root filesystem's total and available space.",
        ],
    },
    "filecraft": {
        "system": "You are a coding assistant in a sandboxed container.",
        "task": "Write a Python script printing the first 10 Fibonacci numbers, run it, confirm it works.",
        "turns": [
            '<|tool_call>call:write_file{content:<|"|>a, b = 0, 1\n'
            "out = []\n"
            "for _ in range(10):\n"
            "    out.append(a)\n"
            "    a, b = b, a + b\n"
            'print(out)\n<|"|>,path:<|"|>fib.py<|"|>}<tool_call|>',
            '<|tool_call>call:shell{command:<|"|>python3 fib.py<|"|>}<tool_call|>',
            '<|tool_call>call:list_dir{path:<|"|>.<|"|>}<tool_call|>',
            "Done. fib.py runs and prints [0, 1, 1, 2, 3, 5, 8, 13, 21, 34].",
        ],
    },
    "datawrangle": {
        "system": "You are a data assistant in a sandboxed container.",
        "task": "Create a JSON file of three users with ages, then report the oldest.",
        "turns": [
            '<|tool_call>call:write_file{'
            'content:<|"|>[{"name":"Ada","age":36},{"name":"Linus","age":54},{"name":"Grace","age":41}]<|"|>,'
            'path:<|"|>users.json<|"|>}<tool_call|>',
            "<|tool_call>call:shell{command:<|\"|>jq -r 'max_by(.age) | .name' users.json<|\"|>}<tool_call|>",
            "The oldest user is Linus (age 54).",
        ],
    },
}

for _wf in SYNTHESIZED.values():
    _wf.setdefault("source", "synthesized (hand-written, native grammar)")


def _load_workflows():
    merged = dict(SYNTHESIZED)
    if _NATIVE_FILE.exists():
        merged.update(json.loads(_NATIVE_FILE.read_text()))  # native wins
    return merged


WORKFLOWS = _load_workflows()
