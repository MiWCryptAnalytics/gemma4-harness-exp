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
from instrument import METRICS, debug, note
from sandbox import Sandbox
from tokens import TOOL_CALL_CLOSE, TURN_CLOSE, clean
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


def build_transformers_engine():
    """Lazily load the real model; return a TransformersEngine.

    Imports torch/transformers only when called, so --dry-run needs neither the
    libraries nor a GPU.
    """
    import os
    # Reduce CUDA fragmentation OOMs as the KV cache grows over a long agent run.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    note(f"loading {model_id} (text engine)...")
    with instrument.Timer() as t:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
    METRICS.model_load_s = t.elapsed
    note(f"model ready in {t.elapsed:.1f}s")

    class TransformersEngine:
        def __call__(self, messages, tools=None, enable_thinking=False):
            prompt = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False,
                add_generation_prompt=True, enable_thinking=enable_thinking,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            n_prompt = inputs["input_ids"].shape[1]
            debug(f"generating (prompt {n_prompt} tok, max_new 1024)...")
            with instrument.Timer() as t:
                out = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    tokenizer=tokenizer,
                    # Stop right after a tool call or at the end of the turn.
                    stop_strings=[TOOL_CALL_CLOSE, TURN_CLOSE],
                )
            gen = out[0][n_prompt:]
            METRICS.record_generation("text-agent", n_prompt, len(gen), t.elapsed)
            # Keep special tokens: we need <|tool_call> / <|"|> to survive.
            return tokenizer.decode(gen, skip_special_tokens=False)

    return TransformersEngine()


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
        reply = engine(messages, tools=schema, enable_thinking=enable_thinking)
        calls = parse_tool_calls(reply)
        debug(f"step {step}: parsed {len(calls)} tool call(s)")

        if not calls:
            answer = clean(reply)
            print(f"\n[Final answer]:\n{answer}")
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
            with instrument.Timer() as t:
                result = dispatch(c["name"], c["arguments"])
            METRICS.record_tool(c["name"], t.elapsed)
            print(f"[Step {step}] result:\n{result}")
            # Feed back a bounded view so a giant build log can't blow up context.
            messages.append({"role": "tool", "name": c["name"],
                             "content": _truncate(result)})

    print(METRICS.summary())

    print(f"\n[Stopped: reached max_steps={max_steps} without a final answer]")
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
    parser.add_argument("--vision", action="store_true",
                        help="Give the agent eyes: load the unified model and the look_at tool.")
    parser.add_argument("--debug", action="store_true",
                        help="Verbose instrumentation: per-step, per-generation, sandbox timings.")
    args = parser.parse_args()

    instrument.reset()
    instrument.set_debug(args.debug)

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
        if args.vision:
            from engine import UnifiedEngine
            import vision_tool  # registers look_at
            import image_tool   # registers create_image
            engine = UnifiedEngine()
            vision_tool.set_engine(engine)
            image_tool.set_engine(engine)
        else:
            engine = build_transformers_engine()

        # Prompts live in external text files so they can be iterated on without
        # touching the code (see prompts/).
        system_file = args.system_prompt_file or args.system_file or load_prompt_path("system.txt")
        system_rules = open(system_file).read().strip()
        if args.vision:
            system_rules += "\n" + open(load_prompt_path("system_vision.txt")).read().strip()
        if args.task_file:
            task = open(args.task_file).read().strip()
        else:
            task = args.task or open(load_prompt_path("default_task.txt")).read().strip()

    answer = None
    with Sandbox(network=args.network, exec_timeout=args.exec_timeout,
                 exec_workspace=args.exec_workspace) as sb:
        try:
            answer = run_agent(system_rules, task, engine, max_steps=args.max_steps,
                               enable_thinking=args.think)
        finally:
            # Export artifacts even if the run errored or hit max_steps, so an
            # external grader still sees whatever the agent managed to produce.
            if args.workspace:
                sb.export_workspace(args.workspace)
                note(f"workspace exported to {args.workspace}")

    engine_kind = "mock" if args.dry_run else ("unified" if args.vision else "text")
    model_label = "mock" if args.dry_run else model_id
    path = METRICS.log_run(model=model_label, engine=engine_kind)
    note(f"metrics written to {path}")

    # Exit code is the eval contract's success signal: 0 iff the agent reached a
    # final answer; non-zero on max_steps (answer is None) so `exit_zero` checks bite.
    sys.exit(0 if answer is not None else 1)


if __name__ == "__main__":
    main()
