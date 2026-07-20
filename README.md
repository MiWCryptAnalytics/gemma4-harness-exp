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

### Transparent MITM proxy (`--network`)

When networking is enabled, the sandbox does **not** reach the internet directly
— every connection is transparently man-in-the-middled so the harness has full
visibility and control over the agent's egress:

- A separate **Squid** container ([`sandbox/squid/`](sandbox/squid/)) — with
  Squid **built from source** in a multi-stage image so it has OpenSSL
  `ssl-bump` + `--enable-linux-netfilter` intercept support — re-signs every TLS
  leaf on the fly with a local **MITM CA**.
- The sandbox trusts **only** that CA: its stock `ca-certificates` bundle is
  stripped to a single cert at build time (verified in the Dockerfile), and
  `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`/`PIP_CERT` point curl,
  wget, Python `requests`, and `pip` at it. Any HTTPS **not** passing through the
  proxy can't validate and **fails closed**.
- Interception is done by having the sandbox **share the proxy's network
  namespace** (`--network container:<proxy>`); the proxy's `iptables` rules
  REDIRECT the agent's `:80`/`:443` into Squid and **DROP all other egress**
  except DNS. Only the proxy is privileged (`NET_ADMIN`) — the sandbox stays
  `--cap-drop ALL`.

One-time setup generates the CA (the private key is gitignored):

```bash
make mitm-ca          # openssl → sandbox/mitm/ca.{crt,key}
make sandbox-build    # builds the sandbox + proxy images (proxy compiles Squid)
make mitm-verify      # proves interception end-to-end (agent curls HTTPS)
```

DNS (port 53) is allowed out un-inspected — an inherent property of transparent
interception (the client resolves the name before the redirect). Non-network and
`--dry-run` runs are unaffected (`--network none`, no proxy).

### Web policy: block / redirect / modify / adapt (`--policy-file`)

Because the proxy already decrypts everything, it can also **act** on it. A hand-
rolled Python **ICAP server** ([sandbox/squid/icap_server.py](sandbox/squid/icap_server.py))
runs inside the proxy and Squid vectors every request through it:

- **REQMOD** — before the origin is contacted: **block** (a synthesized 403 page),
  **redirect** (302), **modify the request** (set/remove headers — e.g. inject
  credentials), or **rewrite the path in place** (regex, same origin). Path rewrites
  + header injection are transparent to the agent — useful for pinning a registry to
  an approved path and adding an auth token. Cross-host offload to a separate mirror
  uses `redirect` (the client follows the 302): an intercepted TLS connection is
  pinned to its original upstream, so an in-place host change can't reroute it.
- **RESPMOD** — after the origin responds: **adapt the body** (regex rewrite /
  inject / redact), content-type scoped, with gzip handled correctly.

Policy is a declarative, hot-reloadable YAML file (first-match-wins rules by
host/URL/method + a `default:` of `allow` or `deny`). Pass one with `--policy-file`;
with none, the baked default is allow-all (identical to pure MITM inspection):

```bash
./venv/bin/python gemma4.py --network --policy-file examples/policy.example.yaml --task "..."
make policy-verify    # proves block + header-pin end-to-end through the agent
```

See [examples/policy.example.yaml](examples/policy.example.yaml) for the schema. If
the ICAP server is unavailable the proxy **fails closed** (Squid errors the request)
rather than leaking un-adapted traffic. The agent (which shares the proxy's netns)
is firewalled off the ICAP port by owner-uid, so it can't reach or tamper with the
policy engine. The engine ([sandbox/squid/policy.py](sandbox/squid/policy.py)) is
pure and host-importable, unit-tested by `test_policy.py`.

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
sandbox.py         Docker sandbox + MITM-proxy lifecycle
sandbox/Dockerfile sandbox image (toolchain, librsvg, scientific Python, MITM-CA-only trust)
sandbox/squid/     MITM proxy: Squid-from-source Dockerfile, squid.conf, entrypoint, ICAP policy server (icap_server.py + policy.py)
sandbox/mitm/      generated MITM CA (gitignored; `make mitm-ca`)
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
