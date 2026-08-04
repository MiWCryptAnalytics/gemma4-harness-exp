"""Structured run trace + the stdout contract the eval suite scrapes.

Two things are checked together on purpose:
  1. trace.jsonl records the run as machine-readable events (what an external
     grader should assert on).
  2. the `[Step N] call:` / `[Step N] result:` / `[Final answer]:` stdout lines
     are UNCHANGED — Gemma4-evals regexes them, so adding the trace must not
     perturb them. This test is the regression lock on that contract.

No GPU, no Docker: a scripted engine plus a tool that needs no sandbox.
"""
import contextlib
import io
import json
import os
import tempfile

import instrument
from gemma4 import run_agent
from tools import REGISTRY, tool, tools_schema


@tool
def echo(text: str):
    """Echo text back (test-only tool; runs on the host, no sandbox needed)."""
    return "echo:" + text


class ScriptedEngine:
    """Replays native-grammar turns, like mockmodel.MockModel."""

    def __init__(self, turns):
        self.turns = turns
        self.calls = 0

    def __call__(self, messages, tools=None, enable_thinking=False):
        reply = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        # Engines report their own generation metrics; mimic that so the trace
        # exercises the record_generation -> trace("generation") path.
        instrument.METRICS.record_generation("test", 10, 5, 0.5)
        return reply


TURNS = [
    '<|tool_call>call:echo{text:<|"|>hello<|"|>}<tool_call|>',
    "All done: the tool echoed hello.",
]

with tempfile.TemporaryDirectory() as tmp:
    trace_path = os.path.join(tmp, "run_trace.jsonl")
    instrument.reset()
    instrument.init_trace(trace_path, model="test", engine="scripted")

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        answer = run_agent("You are a test agent.", "Echo hello.",
                           ScriptedEngine(TURNS), max_steps=4,
                           tools=[REGISTRY["echo"]["schema"]])
    instrument.close_trace()
    out = buf.getvalue()

    events = [json.loads(ln) for ln in open(trace_path) if ln.strip()]
    kinds = [e["event"] for e in events]

    # 1. Event sequence: one step with a tool call, then the answering step.
    assert kinds == ["run_start", "step_start", "generation", "model_output",
                     "tool_call", "tool_result", "step_start", "generation",
                     "model_output", "final_answer"], kinds
    print("event sequence ->", " ".join(kinds))

    # 2. Envelope: every event carries schema version, timestamps, name.
    for e in events:
        assert e["v"] == instrument.TRACE_SCHEMA_VERSION and e["ts"] and "t" in e, e
    assert events[0]["schema_version"] == instrument.TRACE_SCHEMA_VERSION
    print("envelope v/ts/t on all events: OK")

    # 3. A grader can read tool facts structurally — no regex over stdout.
    call = next(e for e in events if e["event"] == "tool_call")
    result = next(e for e in events if e["event"] == "tool_result")
    assert call["name"] == "echo" and call["arguments"] == {"text": "hello"}, call
    assert result["result"] == "echo:hello" and result["step"] == 1, result
    assert next(e for e in events if e["event"] == "final_answer")["answer"] == answer
    print("tool_call/tool_result/final_answer carry structured facts: OK")

    # 4. THE CONTRACT: the eval suite's scrape lines are byte-identical.
    assert "\n[Step 1] call: echo({'text': 'hello'})" in out, out
    assert "\n[Step 1] result:\necho:hello" in out, out
    assert "\n[Final answer]:\nAll done: the tool echoed hello." in out, out
    print("eval stdout contract ([Step N] call/result, [Final answer]): OK")

    # 5. Tracing is off by default — importing/using the harness writes nothing.
    instrument.trace("should_be_dropped", x=1)
    assert len(open(trace_path).read().splitlines()) == len(events)
    print("trace() after close is a no-op: OK")

# 6. max_steps runs emit a terminal event too (so a grader can tell why).
with tempfile.TemporaryDirectory() as tmp:
    trace_path = os.path.join(tmp, "t.jsonl")
    instrument.reset()
    instrument.init_trace(trace_path, model="test")
    with contextlib.redirect_stdout(io.StringIO()):
        answer = run_agent(None, "loop", ScriptedEngine([TURNS[0]]), max_steps=2,
                           tools=[REGISTRY["echo"]["schema"]])
    instrument.close_trace()
    kinds = [json.loads(ln)["event"] for ln in open(trace_path) if ln.strip()]
    assert answer is None and kinds[-1] == "max_steps", kinds
    print("max_steps event on exhaustion: OK")

# 7. An unwritable location costs the run its trace, not the run itself.
instrument.reset()
assert instrument.init_trace("/proc/nonexistent-dir/t.jsonl", model="test") is None
with contextlib.redirect_stdout(io.StringIO()) as buf:
    answer = run_agent(None, "Echo hello.", ScriptedEngine(TURNS), max_steps=4,
                       tools=[REGISTRY["echo"]["schema"]])
assert answer == "All done: the tool echoed hello.", answer
assert "[Step 1] call: echo(" in buf.getvalue()
print("unwritable trace path -> run still completes: OK")

del REGISTRY["echo"]  # keep the global registry clean for other tests
assert "echo" not in [s["function"]["name"] for s in tools_schema()]

print("\nall trace tests passed")
