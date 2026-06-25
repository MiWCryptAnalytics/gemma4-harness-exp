# Gemma 4 Agentic Harness

A compact, sandboxed agent harness for **`google/gemma-4-12b-it`** that drives the
model through its **native reserved-token tool grammar** — and gives it a full
sensorium of tools: a shell, a Python data-science brain, eyes (vision), the
ability to draw, and the ability to compose music. One model, all text-driven,
everything isolated in Docker.

> Built as an exploration: every capability here was added by *running the model
> for real* and watching what it actually does. Several non-obvious findings
> (the model's native tool format, sandbox `noexec`, `docker cp` vs tmpfs) came
> straight out of those runs — see the `probe_*.py` scripts.

## The agent's tools

| Power | Tool(s) | Notes |
|-------|---------|-------|
| 🖐 Hands | `shell`, `read_file`, `write_file`, `list_dir` | run anything in the sandbox |
| 🏗 Build | (uses `shell` + toolchain in the image) | e.g. downloads & compiles nginx from source |
| 🧠 Brain | `run_python` | numpy / sympy / pandas / matplotlib |
| 👁 Eyes | `look_at` | sees images via Gemma's own vision tower |
| 🎨 Draw | `create_image` | composes SVG, renders it, self-scores the result |
| 🎵 Voice | `compose_music` | writes ABC notation → synthesizes a WAV |

## The core idea: native tool calling

Gemma 4 has **reserved control tokens** for tool use (verified in
[`probe_tokenizer.py`](probe_tokenizer.py)):

```
<|tool_call>call:NAME{key:<|"|>value<|"|>,...}<tool_call|>
```

The `<|"|>` token delimits string values, so arguments with newlines, quotes, or
commas are never ambiguous. The harness declares tools via the model's chat
template and parses this grammar directly ([`tools.py`](tools.py)) — which
eliminated a whole class of parsing failures that an ad-hoc markdown format
suffered from.

## Requirements

- **NVIDIA GPU, ~24 GB VRAM** — the 12B model at bf16 (less works but spills to
  CPU and is slow)
- **Docker** — all tool execution runs in a container
- **Python 3.12+** (developed on 3.14)
- A **Hugging Face account with access to the gated Gemma weights**

The no-GPU paths (`make test`, `make dry-run`) need neither a GPU nor the model.

## Setup

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
make doctor          # preflight: checks deps, Docker, GPU, and the model
```

### Getting the model

`google/gemma-4-12b-it` is **gated** and is **not** included in this repo. Accept
the [Gemma Terms of Use](https://ai.google.dev/gemma/terms) on the model page,
then authenticate so it downloads into your HF cache on first run:

```bash
./venv/bin/huggingface-cli login
```

To compare another model, set `GEMMA_MODEL_ID=<org/model>` — but note the native
tool-call grammar and chat template are tuned to Gemma 4, so other models may
need template adjustments.

The sandbox Docker image builds automatically on first run (or `make sandbox-build`).

## Usage

Everything runs through the [`Makefile`](Makefile) (`make help` lists targets):

```bash
make demo      # the grand variety show — every tool in one run (GPU)
make nginx     # agent downloads + compiles nginx from source
make chart     # agent computes, plots, and SEES a chart
make music     # agent composes ABC music → WAV
make image     # quality-gated image agent
make sysinfo   # agent inspects its sandbox

make test      # fast no-GPU correctness tests
make dry-run   # replay a recorded workflow with no GPU at all
make metrics   # tabulate recorded run metrics
make clean     # remove generated artifacts
```

Or drive `gemma4.py` directly:

```bash
./venv/bin/python gemma4.py --vision --debug --task "Draw a fox, then look at it."
./venv/bin/python gemma4.py --network --exec-workspace --task "Compile nginx ..."
```

Useful flags: `--vision` (eyes + draw), `--network`, `--exec-workspace` (run
compiled binaries), `--exec-timeout`, `--max-steps`, `--task`/`--task-file`,
`--system-file`/`--system-prompt-file`, `--workspace`, `--debug`, `--dry-run`.

### Prompts

The agent's system rules and tasks live as editable text in
[`prompts/`](prompts/) — `system.txt` (base rules), `system_vision.txt` (appended
under `--vision`), `default_task.txt`, and `demo.txt` — so you can iterate on them
without touching the code. Override per-run with `--system-file` / `--task-file`.

## Sandbox & safety

All tool execution happens inside a disposable Docker container
([`sandbox.py`](sandbox.py), [`sandbox/Dockerfile`](sandbox/Dockerfile)):

- no host bind mounts · `--network none` by default · `--read-only` root FS
- non-root user · `--cap-drop ALL` · `--pids/--memory/--cpus` limits
- writable **tmpfs** workspace, `noexec` unless `--exec-workspace`

Untrusted, model-authored SVG is sanitized (text/script/external refs stripped)
*and* rasterized inside the sandbox, so it can't smuggle instructions into the
vision-scoring loop or exploit the renderer.

## Instrumentation

Every run reports tokens/sec per generation and a summary (model-load time,
throughput by type, tool time, wall clock), and writes a
`metrics/<timestamp>_<model>.json` for cross-run comparison. Set a different
model with `GEMMA_MODEL_ID=…` to compare (note: the tool grammar is tuned to
Gemma 4, so other models may need template tweaks).

## Evaluation harness

The instrumentation above measures *speed*; a separate tool measures *task
success*. The harness exposes a small contract so an external eval driver — the
companion [`Gemma4-evals`](../Gemma4-evals) project — can run a suite of agentic
tasks against it and grade what the agent actually produced:

- `--workspace <hostdir>` — after the run, the sandbox's tmpfs `/workspace` is
  exported (tar-over-`docker exec`, same trick as binary reads) to this host
  directory, so file/shell checks and an LLM vision-judge can inspect artifacts
  like `chart.png`. Export happens even if the run errors or hits `--max-steps`.
- `--system-prompt-file <file>` — the system prompt under evaluation (alias of
  `--system-file`); this is the prompt-tuning target the driver injects.
- **Exit code** — `0` when the agent reached a final answer, non-zero when it hit
  `--max-steps` or crashed, so the eval's `exit_zero` check is meaningful.

The eval driver shells out per its own `harness.yaml`; point that file's
`command` at this `gemma4.py` (absolute path — the driver runs each task in its
own working directory, and the harness is cwd-independent). The grader and task
suite live in `Gemma4-evals`.

## Repository layout

```
gemma4.py          native tool-calling agent loop + CLI
engine.py          UnifiedEngine (one model: text tool-calling + vision)
tokens.py          reserved control-token constants
tools.py           tool registry, schema gen, native <|tool_call> parser, sandbox tools
sandbox.py         Docker sandbox lifecycle
sandbox/Dockerfile sandbox image (toolchain, librsvg, scientific Python)
instrument.py      metrics + debug instrumentation
svg_studio.py      SVG generate→render→self-critique→refine loop + sanitizer
image_tool.py      create_image (quality-gated)
vision_tool.py     look_at
music.py           ABC-notation → WAV synthesizer
multimodal.py      multimodal message construction + standalone vision engine
mockmodel.py       no-GPU workflow replay
generate_workflows.py  records native model workflows for dry-run
image_agent.py     standalone quality-gated image agent
prompts/           editable system rules + tasks (system.txt, demo.txt, ...)
probe_*.py         the investigations behind the design decisions
test_*.py          no-GPU correctness tests
```

## License

MIT — see [LICENSE](LICENSE). The **Gemma model** itself is governed by Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), which you must accept
separately; this repository contains no model weights.
