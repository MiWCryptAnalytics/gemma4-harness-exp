"""Probe: does carrying the KV cache across agent steps change the output?

Runs the same scripted two-step conversation twice — once with cache reuse, once
without — and compares the model's replies token for token. Reuse is only ever
taken on an exact prefix extension, so the outputs MUST match; this probe is the
empirical proof, plus a speedup measurement.

    ./venv/bin/python probe_kv_cache.py [--quantize 8bit]

Needs the GPU and the real model (unlike test_engine.py, which covers the pure
prefix logic offline).
"""
import argparse
import json
import time

import instrument
from engine import UnifiedEngine
from tools import tools_schema

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None)
parser.add_argument("--vision", action="store_true",
                    help="Probe the unified vision+text engine instead of text-only.")
args = parser.parse_args()

instrument.set_debug(True)

# A two-step conversation shaped exactly like run_agent's: the second turn's
# prompt re-renders the first turn plus a tool response.
STEP1 = [
    {"role": "system", "content": "You are a Linux assistant in a sandbox."},
    {"role": "user", "content": "What kernel is this machine running? Use your tools."},
]
STEP2 = STEP1 + [
    {"role": "assistant", "content": "",
     "tool_calls": [{"type": "function",
                     "function": {"name": "shell", "arguments": {"command": "uname -srm"}}}]},
    {"role": "tool", "name": "shell", "content": "Linux 6.8.0-45-generic x86_64"},
]

SCHEMA = tools_schema()


def run(kv_reuse):
    engine = UnifiedEngine(quantize=args.quantize, vision=args.vision,
                           kv_reuse=kv_reuse)
    replies, timings = [], []
    for messages in (STEP1, STEP2):
        t0 = time.perf_counter()
        replies.append(engine(messages, tools=SCHEMA))
        timings.append(time.perf_counter() - t0)
    del engine
    import torch
    torch.cuda.empty_cache()
    return replies, timings


print("\n=== run A: kv_reuse=False (rebuild every step) ===")
base_replies, base_times = run(kv_reuse=False)
print("\n=== run B: kv_reuse=True (carry the cache) ===")
kv_replies, kv_times = run(kv_reuse=True)

print("\n" + "=" * 64)
identical = base_replies == kv_replies
for i, (a, b) in enumerate(zip(base_replies, kv_replies), 1):
    same = "IDENTICAL" if a == b else "DIFFERENT"
    print(f"step {i}: {same}  ({base_times[i-1]:.1f}s uncached vs {kv_times[i-1]:.1f}s cached)")
    if a != b:
        print(f"  uncached: {a[:200]!r}")
        print(f"  cached:   {b[:200]!r}")

# Step 2 is the one that can reuse (step 1 has nothing cached yet).
speedup = base_times[1] / kv_times[1] if kv_times[1] > 0 else 0.0
print(f"\nstep-2 speedup from cache reuse: {speedup:.2f}x")
print("VERDICT:", "PASS — cache reuse is output-identical" if identical
      else "FAIL — cache reuse changed the output; investigate before trusting it")

with open("kv_cache_probe.json", "w") as fh:
    json.dump({"quantize": args.quantize, "vision": args.vision,
               "identical": identical, "uncached_s": base_times,
               "cached_s": kv_times, "step2_speedup": round(speedup, 3),
               "replies_uncached": base_replies, "replies_cached": kv_replies},
              fh, indent=2)
print("wrote kv_cache_probe.json")
raise SystemExit(0 if identical else 1)
