"""Lightweight instrumentation: debug notes + throughput metrics.

Shows what the harness is doing and how fast — tokens/sec per generation, model
load time, tool and sandbox-command timings — so runs are comparable across
models. A concise throughput line prints per generation; verbose step/sandbox
notes appear under --debug; an aggregate summary prints at the end of a run.
"""

import datetime
import json
import os
import re
import time

_DEBUG = False

_DIM = "\033[90m"
_CYAN = "\033[36m"
_RST = "\033[0m"


def set_debug(on):
    global _DEBUG
    _DEBUG = bool(on)


def debug(msg):
    """Verbose note, only shown under --debug."""
    if _DEBUG:
        print(f"{_DIM}  [debug] {msg}{_RST}")


def note(msg):
    """Normal-level harness note (always shown)."""
    print(f"{_DIM}  [harness] {msg}{_RST}")


class Timer:
    """Context manager returning elapsed seconds via .elapsed."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0


class Metrics:
    def __init__(self):
        self.model_load_s = None
        self.generations = []   # {label, prompt_tokens, gen_tokens, seconds, tps}
        self.tools = []         # {name, seconds}
        self.t_start = time.perf_counter()

    def record_generation(self, label, prompt_tokens, gen_tokens, seconds):
        tps = gen_tokens / seconds if seconds > 0 else 0.0
        self.generations.append(dict(label=label, prompt_tokens=prompt_tokens,
                                     gen_tokens=gen_tokens, seconds=seconds, tps=tps))
        # The headline metric — always visible, kept to one dim line.
        print(f"{_DIM}  ⚡ {label}: {gen_tokens} tok in {seconds:.1f}s = "
              f"{_CYAN}{tps:.1f} tok/s{_RST}{_DIM} (prompt {prompt_tokens} tok){_RST}")

    def record_tool(self, name, seconds):
        self.tools.append(dict(name=name, seconds=seconds))
        debug(f"tool {name} took {seconds:.2f}s")

    def summary(self):
        gen_tok = sum(g["gen_tokens"] for g in self.generations)
        gen_s = sum(g["seconds"] for g in self.generations)
        tool_s = sum(t["seconds"] for t in self.tools)
        wall = time.perf_counter() - self.t_start
        avg = gen_tok / gen_s if gen_s > 0 else 0.0
        all_tps = [g["tps"] for g in self.generations]

        bar = "─" * 58
        out = ["", bar, f"{_CYAN}Run metrics{_RST}", bar]
        if self.model_load_s is not None:
            out.append(f"  model load        {self.model_load_s:7.1f}s")
        out.append(f"  generations       {len(self.generations)}")
        out.append(f"  tokens generated  {gen_tok}")
        if all_tps:
            out.append(f"  throughput        {avg:.1f} tok/s avg  "
                       f"({min(all_tps):.1f}-{max(all_tps):.1f} range)")
        out.append(f"  generation time   {gen_s:7.1f}s")
        out.append(f"  tool calls        {len(self.tools)}  ({tool_s:.1f}s)")
        out.append(f"  wall clock        {wall:7.1f}s")

        # Per-type breakdown (text-agent vs vision vs svg, etc.) — handy for
        # comparing where a model spends time.
        by_label = {}
        for g in self.generations:
            d = by_label.setdefault(g["label"], [0, 0.0])
            d[0] += g["gen_tokens"]
            d[1] += g["seconds"]
        if len(by_label) > 1:
            out.append("  by type:")
            for label, (tok, sec) in sorted(by_label.items()):
                tps = tok / sec if sec > 0 else 0.0
                out.append(f"    {label:<14} {tok:>6} tok  {sec:6.1f}s  {tps:6.1f} tok/s")
        out.append(bar)
        return "\n".join(out)

    def as_dict(self):
        """Aggregate stats as a plain dict (for JSON logging / comparison)."""
        gen_tok = sum(g["gen_tokens"] for g in self.generations)
        gen_s = sum(g["seconds"] for g in self.generations)
        tool_s = sum(t["seconds"] for t in self.tools)
        wall = time.perf_counter() - self.t_start
        by = {}
        for g in self.generations:
            d = by.setdefault(g["label"], {"tokens": 0, "seconds": 0.0})
            d["tokens"] += g["gen_tokens"]
            d["seconds"] += g["seconds"]
        for d in by.values():
            d["seconds"] = round(d["seconds"], 2)
            d["tok_s"] = round(d["tokens"] / d["seconds"], 1) if d["seconds"] > 0 else 0.0
        return {
            "model_load_s": round(self.model_load_s, 2) if self.model_load_s is not None else None,
            "generations": len(self.generations),
            "tokens": gen_tok,
            "gen_seconds": round(gen_s, 2),
            "avg_tok_s": round(gen_tok / gen_s, 2) if gen_s > 0 else 0.0,
            "tool_calls": len(self.tools),
            "tool_seconds": round(tool_s, 2),
            "wall_s": round(wall, 2),
            "by_type": by,
        }

    def log_run(self, model, outdir="metrics", **extra):
        """Write this run's metrics to a unique metrics/<timestamp>_<model>.json.

        One file per run (named by time + model) so results from different models
        sit side by side for comparison. Includes the per-generation detail too.
        """
        os.makedirs(outdir, exist_ok=True)
        now = datetime.datetime.now()
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model)).strip("-") or "model"
        # Sub-second precision so two runs in the same second don't collide.
        stamp = now.strftime("%Y%m%dT%H%M%S_") + f"{now.microsecond // 1000:03d}"
        path = os.path.join(outdir, f"{stamp}_{safe_model}.json")
        record = {
            "ts": now.isoformat(timespec="seconds"),
            "model": model,
            **extra,
            **self.as_dict(),
            "generations_detail": self.generations,
        }
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        return path


METRICS = Metrics()


def reset():
    """Start a fresh metrics collection (call at the top of a run)."""
    global METRICS
    METRICS = Metrics()
    return METRICS
