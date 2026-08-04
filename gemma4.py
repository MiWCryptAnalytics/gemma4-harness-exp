"""Gemma 4 agentic harness — native reserved-token tool calling.

The model declares/calls tools through Gemma 4's own control tokens (see
tokens.py and probe_*.py for how this was reverse-engineered):

    declare:  apply_chat_template(tools=[...]) -> <|tool>declaration:...<tool|>
    call:     model emits  <|tool_call>call:NAME{...}<tool_call|>
    respond:  we feed a role:tool message -> <|tool_response>...<tool_response|>

All tool side effects run inside the Docker sandbox (sandbox.py). Generation
stops at <tool_call|> (a call to execute) or <turn|> (a final answer).
"""

import argparse
import os
import sys
import time

import instrument
from instrument import METRICS, debug, note, trace
from sandbox import Sandbox
from tokens import clean, extract_channels
from tools import REGISTRY, dispatch, parse_tool_calls, tools_schema

# Override with GEMMA_MODEL_ID to compare other models (note: the native
# tool-call grammar and chat template are tuned to Gemma 4).
model_id = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12b-it")

# System rules and tasks are kept as editable text in prompts/ rather than
# hard-coded here, so they can be iterated on without changing the code.
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


def load_prompt_path(name):
    return os.path.join(PROMPTS_DIR, name)

# Cap how much of a tool's output is fed back into context. Build commands
# (make/configure/tar -v) emit thousands of lines; replaying them verbatim wastes
# context and can OOM the KV cache. Errors usually sit at the END, so we keep more
# of the tail than the head.
_TOOL_OUTPUT_HEAD = 800
_TOOL_OUTPUT_TAIL = 3200


def _truncate(text):
    text = str(text)
    if len(text) <= _TOOL_OUTPUT_HEAD + _TOOL_OUTPUT_TAIL:
        return text
    omitted = len(text) - _TOOL_OUTPUT_HEAD - _TOOL_OUTPUT_TAIL
    return (text[:_TOOL_OUTPUT_HEAD]
            + f"\n... [{omitted} chars omitted] ...\n"
            + text[-_TOOL_OUTPUT_TAIL:])


def _summarize_net_audit(text):
    """Condense the proxy's audit JSONL into counts for the metrics record."""
    import json
    requests = blocked = redirected = rewritten = 0
    hosts = set()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        action = rec.get("action")
        if rec.get("phase") == "reqmod":
            requests += 1
            hosts.add(rec.get("host", ""))
            blocked += action == "block"
            redirected += action == "redirect"
        elif action == "rewrite-body":
            rewritten += 1
    return {"requests": requests, "blocked": blocked, "redirected": redirected,
            "bodies_rewritten": rewritten, "hosts": len(hosts)}


def build_transformers_engine(quantize=None, kv_reuse=False, stream=False):
    """Lazily load the real model, text-only; return a UnifiedEngine.

    Imports torch/transformers only when called (via engine), so --dry-run needs
    neither the libraries nor a GPU. `quantize` (None | '4bit' | '8bit') is
    opt-in — see engine.model_load_kwargs for the fidelity tradeoff. vision=False
    loads AutoModelForCausalLM, so the vision tower costs no VRAM on this path
    while the generation logic stays shared with the --vision engine.
    """
    from engine import UnifiedEngine
    return UnifiedEngine(model_id=model_id, quantize=quantize, vision=False,
                         kv_reuse=kv_reuse, stream=stream)


def run_agent(system_instruction, user_prompt, engine, max_steps=8,
              enable_thinking=False, tools=None):
    """Drive a native tool-using conversation until a final answer.

    `engine(messages, tools=, enable_thinking=)` returns the model's raw reply
    text (control tokens intact). `tools` is the schema list to expose (defaults
    to every registered tool). Returns the final prose answer.
    """
    schema = tools if tools is not None else tools_schema()
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": user_prompt})

    for step in range(1, max_steps + 1):
        debug(f"step {step}/{max_steps}: {len(messages)} messages in context")
        trace("step_start", step=step, max_steps=max_steps, n_messages=len(messages))
        reply = engine(messages, tools=schema, enable_thinking=enable_thinking)
        calls = parse_tool_calls(reply)
        debug(f"step {step}: parsed {len(calls)} tool call(s)")
        trace("model_output", step=step, raw=reply, raw_len=len(reply),
              n_calls=len(calls))
        # clean() drops the reasoning channel; capture it first so a --think run
        # keeps its reasoning in the trace (full text there, a snippet here).
        thinking = extract_channels(reply)
        if thinking:
            trace("thinking", step=step, texts=thinking,
                  chars=sum(len(t) for t in thinking))
            debug(f"think: {thinking[0][:200]}")

        if not calls:
            answer = clean(reply)
            print(f"\n[Final answer]:\n{answer}")
            trace("final_answer", step=step, answer=answer, chars=len(answer))
            print(METRICS.summary())
            return answer

        # Record the assistant's tool call(s) as structured history; the chat
        # template re-serializes them into the native grammar next turn.
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls
            ],
        })

        for c in calls:
            print(f"\n[Step {step}] call: {c['name']}({c['arguments']})")
            trace("tool_call", step=step, name=c["name"], arguments=c["arguments"])
            with instrument.Timer() as t:
                result = dispatch(c["name"], c["arguments"])
            METRICS.record_tool(c["name"], t.elapsed)
            print(f"[Step {step}] result:\n{result}")
            # Feed back a bounded view so a giant build log can't blow up context.
            fed = _truncate(result)
            trace("tool_result", step=step, name=c["name"], result=fed,
                  result_len=len(str(result)), seconds=round(t.elapsed, 3))
            messages.append({"role": "tool", "name": c["name"], "content": fed})

    print(METRICS.summary())

    print(f"\n[Stopped: reached max_steps={max_steps} without a final answer]")
    trace("max_steps", max_steps=max_steps)
    return None


def main():
    parser = argparse.ArgumentParser(description="Gemma 4 native-tool-calling harness")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replay a recorded workflow instead of loading the GPU model.")
    parser.add_argument("--workflow", default="sysinfo",
                        help="Which workflow to replay in --dry-run mode.")
    parser.add_argument("--network", action="store_true",
                        help="Allow network access inside the sandbox.")
    parser.add_argument("--think", action="store_true",
                        help="Enable the model's reasoning channel.")
    parser.add_argument("--task", default=None,
                        help="Override the user task (real-model mode).")
    parser.add_argument("--task-file", default=None,
                        help="Read the task from a file (real-model mode).")
    parser.add_argument("--system-file", default=None,
                        help="System-rules file (default: prompts/system.txt).")
    parser.add_argument("--system-prompt-file", default=None,
                        help="System prompt under evaluation; alias of --system-file "
                             "(the tuning target an external eval driver injects).")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max tool-call rounds before giving up.")
    parser.add_argument("--exec-timeout", type=int, default=60,
                        help="Per-command timeout (s) inside the sandbox; raise for compiles.")
    parser.add_argument("--exec-workspace", action="store_true",
                        help="Allow executing files from the workspace (needed to build/run compiled code).")
    parser.add_argument("--workspace", default=None,
                        help="Host directory to export the sandbox /workspace into after the run "
                             "(so an external eval driver can grade the artifacts).")
    parser.add_argument("--policy-file", default=None,
                        help="Host YAML web-policy file (block/redirect/modify/adapt), hot-reloaded "
                             "into the MITM proxy. Requires --network.")
    parser.add_argument("--vision", action="store_true",
                        help="Give the agent eyes and ears: load the unified model plus the "
                             "look_at, create_image and listen tools.")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Opt-in weight quantization (speed over fidelity). Default is "
                             "full bf16: 4-bit is known to emit malformed SVG/XML, breaking "
                             "structured tool output. 8bit (~13GB) fits a 24GB card on-GPU; "
                             "4bit (~7GB) is fastest but least faithful.")
    parser.add_argument("--stream", action="store_true",
                        help="Stream generated tokens live to stderr as they arrive.")
    parser.add_argument("--kv-reuse", action="store_true",
                        help="Carry the KV cache across agent steps. Off by default: "
                             "measured to never hit on Gemma 4's chat template "
                             "(see engine.KV_REUSE_NOTE), so it would only pin VRAM.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose instrumentation: per-step, per-generation, sandbox timings.")
    args = parser.parse_args()

    # A web policy only has effect on the MITM-proxied network path.
    if args.policy_file and not args.network:
        parser.error("--policy-file requires --network (the policy runs in the MITM proxy)")

    instrument.reset()
    instrument.set_debug(args.debug)

    # One stamp for every artifact this run writes (metrics, trace, net audit).
    stamp = instrument.run_stamp()
    engine_kind = "mock" if args.dry_run else ("unified" if args.vision else "text")
    model_label = "mock" if args.dry_run else model_id
    trace_path = instrument.init_trace(
        os.path.join("metrics", f"{stamp}_trace.jsonl"),
        model=model_label, engine=engine_kind, quantize=args.quantize,
        network=args.network, vision=args.vision, think=args.think,
        max_steps=args.max_steps, policy_file=args.policy_file,
        exec_workspace=args.exec_workspace, exec_timeout=args.exec_timeout,
    )

    if args.dry_run:
        from mockmodel import MockModel, WORKFLOWS
        if args.workflow not in WORKFLOWS:
            parser.error(f"unknown workflow '{args.workflow}'. Choices: {', '.join(WORKFLOWS)}")
        wf = WORKFLOWS[args.workflow]
        engine = MockModel(wf["turns"])
        system_rules, task = wf["system"], wf["task"]
        print(f"[dry-run] replaying '{args.workflow}' "
              f"({len(wf['turns'])} turns, source: {wf.get('source', 'unknown')}), no GPU.")
    else:
        kv_reuse = args.kv_reuse
        if args.network:
            import web_tool  # noqa: F401  (registers browse)
        if args.vision:
            from engine import UnifiedEngine
            import vision_tool  # registers look_at
            import image_tool   # registers create_image
            import audio_tool   # registers listen (hears via the vision tower)
            engine = UnifiedEngine(quantize=args.quantize, kv_reuse=kv_reuse,
                                   stream=args.stream)
            vision_tool.set_engine(engine)
            image_tool.set_engine(engine)
            audio_tool.set_engine(engine)
        else:
            engine = build_transformers_engine(quantize=args.quantize,
                                               kv_reuse=kv_reuse, stream=args.stream)

        # Prompts live in external text files so they can be iterated on without
        # touching the code (see prompts/).
        system_file = args.system_prompt_file or args.system_file or load_prompt_path("system.txt")
        system_rules = open(system_file).read().strip()
        if args.vision:
            system_rules += "\n" + open(load_prompt_path("system_vision.txt")).read().strip()
        if args.network:
            system_rules += "\n" + open(load_prompt_path("system_network.txt")).read().strip()
        if args.task_file:
            task = open(args.task_file).read().strip()
        else:
            task = args.task or open(load_prompt_path("default_task.txt")).read().strip()

    answer = None
    with Sandbox(network=args.network, exec_timeout=args.exec_timeout,
                 exec_workspace=args.exec_workspace,
                 policy_file=args.policy_file) as sb:
        try:
            answer = run_agent(system_rules, task, engine, max_steps=args.max_steps,
                               enable_thinking=args.think)
        except Exception as exc:
            trace("error", type=type(exc).__name__, message=str(exc))
            raise
        finally:
            # Export artifacts even if the run errored or hit max_steps, so an
            # external grader still sees whatever the agent managed to produce.
            # The trace deliberately stays in metrics/: anything left in the
            # workspace would show up in an eval's artifact globs.
            if args.workspace:
                sb.export_workspace(args.workspace)
                note(f"workspace exported to {args.workspace}")

    extra = {"engine": engine_kind}
    if trace_path:
        extra["trace"] = trace_path
    # The MITM proxy records every request it decided on; keep it beside the
    # run's other artifacts so a grader can assert on network behavior.
    if sb.net_audit:
        audit_path = os.path.join("metrics", f"{stamp}_netaudit.jsonl")
        with open(audit_path, "w") as fh:
            fh.write(sb.net_audit)
        summary = _summarize_net_audit(sb.net_audit)
        note(f"network audit written to {audit_path} "
             f"({summary['requests']} requests, {summary['blocked']} blocked)")
        trace("net_audit_summary", path=audit_path, **summary)
        extra["net_audit"] = summary

    path = METRICS.log_run(model=model_label, stamp=stamp, **extra)
    note(f"metrics written to {path}")
    if trace_path:
        note(f"trace written to {trace_path}")
    trace("run_end", answered=answer is not None, metrics=path)
    instrument.close_trace()

    # Exit code is the eval contract's success signal: 0 iff the agent reached a
    # final answer; non-zero on max_steps (answer is None) so `exit_zero` checks bite.
    sys.exit(0 if answer is not None else 1)


if __name__ == "__main__":
    main()
