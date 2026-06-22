"""Tool registry, JSON-schema generation, and native tool-call parsing.

Tools are plain functions registered with @tool. From each signature we build an
OpenAI-style schema that apply_chat_template(tools=...) renders into Gemma's
native `<|tool>declaration:...<tool|>` block. When the model emits
`<|tool_call>call:NAME{...}<tool_call|>`, parse_tool_calls() reads it back.

Parsing is robust by construction: string values are wrapped in the reserved
`<|"|>` delimiter token, which cannot appear inside content — so raw newlines,
commas, and quotes in arguments are never ambiguous (the failure mode the old
markdown/eval parser suffered from).

Every env tool executes inside the Docker sandbox (sandbox.py), never on the host.
"""

import inspect

from sandbox import get_active
from tokens import STR_DELIM

# name -> {"func": callable, "schema": dict}
REGISTRY = {}

_SHELL_DENYLIST = ("rm -rf /", "mkfs", "dd if=", ":(){", "shutdown", "reboot")

# Python annotation -> JSON-schema type.
_JSON_TYPE = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}


def tool(func):
    """Register a function as a tool; its signature becomes the JSON schema.

    Annotate params (e.g. path: str) to type them; the first docstring line is
    the tool description. Params without a default are marked required.
    """
    sig = inspect.signature(func)
    doc = (inspect.getdoc(func) or "").strip().split("\n")[0]
    props, required = {}, []
    for name, p in sig.parameters.items():
        annot = p.annotation if p.annotation is not inspect.Parameter.empty else str
        props[name] = {"type": _JSON_TYPE.get(annot, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    REGISTRY[func.__name__] = {
        "func": func,
        "schema": {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        },
    }
    return func


def tools_schema():
    """Return the list of tool schemas to pass to apply_chat_template(tools=...)."""
    return [entry["schema"] for entry in REGISTRY.values()]


def dispatch(name, arguments):
    """Run a registered tool by name with a kwargs dict; return result as str."""
    if name not in REGISTRY:
        return f"Error: unknown tool '{name}'. Available: {', '.join(REGISTRY)}"
    try:
        return str(REGISTRY[name]["func"](**arguments))
    except Exception as exc:  # surface failures so the model can adapt
        return f"Error calling {name}: {type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# Native tool-call parsing:  <|tool_call>call:NAME{k:v,...}<tool_call|>
# A small recursive-descent reader over the {...} body. String values/keys are
# delimited by the reserved STR_DELIM token; everything else is a bare scalar,
# nested {object}, or [array].
# --------------------------------------------------------------------------

def _skip_ws(s, i):
    while i < len(s) and s[i] in " \n\t":
        i += 1
    return i


def _coerce(token):
    token = token.strip()
    if token in ("true", "false"):
        return token == "true"
    if token in ("null", "None", ""):
        return None
    for cast in (int, float):
        try:
            return cast(token)
        except ValueError:
            pass
    return token  # bare/unquoted string


def _read_delimited(s, i):
    """Read a STR_DELIM-wrapped string starting at s[i]; return (value, next)."""
    i += len(STR_DELIM)
    end = s.find(STR_DELIM, i)
    if end == -1:  # tolerate a missing closing delimiter at end of stream
        return s[i:], len(s)
    return s[i:end], end + len(STR_DELIM)


def _parse_value(s, i):
    i = _skip_ws(s, i)
    if s.startswith(STR_DELIM, i):
        return _read_delimited(s, i)
    if i < len(s) and s[i] == "{":
        return _parse_map(s, i)
    if i < len(s) and s[i] == "[":
        return _parse_list(s, i)
    j = i
    while j < len(s) and s[j] not in ",}]":
        j += 1
    return _coerce(s[i:j]), j


def _parse_key(s, i):
    i = _skip_ws(s, i)
    if s.startswith(STR_DELIM, i):
        return _read_delimited(s, i)
    j = i
    while j < len(s) and s[j] != ":":
        j += 1
    return s[i:j].strip(), j


def _parse_map(s, i):
    i = _skip_ws(s, i + 1)  # skip '{'
    obj = {}
    if i < len(s) and s[i] == "}":
        return obj, i + 1
    while i < len(s):
        key, i = _parse_key(s, i)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ":":
            i += 1
        value, i = _parse_value(s, i)
        obj[key] = value
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i += 1
            continue
        if i < len(s) and s[i] == "}":
            return obj, i + 1
        break
    return obj, i


def _parse_list(s, i):
    i = _skip_ws(s, i + 1)  # skip '['
    arr = []
    if i < len(s) and s[i] == "]":
        return arr, i + 1
    while i < len(s):
        value, i = _parse_value(s, i)
        arr.append(value)
        i = _skip_ws(s, i)
        if i < len(s) and s[i] == ",":
            i += 1
            continue
        if i < len(s) and s[i] == "]":
            return arr, i + 1
        break
    return arr, i


def parse_tool_calls(text):
    """Extract every <|tool_call>call:NAME{...}<tool_call|> from model output.

    Returns a list of {"name": str, "arguments": dict}. Empty if none present.
    """
    from tokens import TOOL_CALL_OPEN, TOOL_CALL_CLOSE
    calls, i = [], 0
    while True:
        start = text.find(TOOL_CALL_OPEN, i)
        if start == -1:
            break
        start += len(TOOL_CALL_OPEN)
        close = text.find(TOOL_CALL_CLOSE, start)
        body = text[start: close if close != -1 else len(text)].strip()
        i = (close + len(TOOL_CALL_CLOSE)) if close != -1 else len(text)

        if body.startswith("call:"):
            body = body[len("call:"):]
        brace = body.find("{")
        if brace == -1:
            continue
        name = body[:brace].strip()
        arguments, _ = _parse_map(body, brace)
        calls.append({"name": name, "arguments": arguments})
    return calls


# --------------------------------------------------------------------------
# Built-in environment tools (all execute inside the sandbox container)
# --------------------------------------------------------------------------

@tool
def shell(command: str):
    """Run a shell command in the sandbox and return its stdout/stderr."""
    if any(bad in command for bad in _SHELL_DENYLIST):
        return f"Refused: command matched the safety denylist ({command!r})."
    return get_active().run(command)


@tool
def read_file(path: str):
    """Read and return the text contents of a file in the sandbox."""
    import shlex
    return get_active().run(f"cat -- {shlex.quote(path)}")


@tool
def write_file(path: str, content: str):
    """Write text content to a file in the sandbox (parent dirs auto-created)."""
    import shlex
    q = shlex.quote(path)
    sandbox = get_active()
    mkdir = sandbox.run(f'mkdir -p -- "$(dirname {q})"')
    if mkdir.exit_code != 0:
        return mkdir
    result = sandbox.run(f"cat > {q}", stdin=content)
    return f"Wrote {len(content)} chars to {path}" if result.exit_code == 0 else result


@tool
def list_dir(path: str = "."):
    """List the entries in a sandbox directory."""
    import shlex
    return get_active().run(f"ls -1Ap -- {shlex.quote(path)}")


@tool
def run_python(code: str):
    """Run Python 3 in the sandbox (numpy, sympy, pandas, matplotlib available) and return its output."""
    sandbox = get_active()
    # Pass the code via stdin so multi-line scripts need no shell escaping; run
    # the file with python3 (reads the script — no exec bit / exec-workspace needed).
    wrote = sandbox.run("cat > /workspace/_snippet.py", stdin=code)
    if wrote.exit_code != 0:
        return wrote
    return sandbox.run("python3 /workspace/_snippet.py")


@tool
def compose_music(abc: str, path: str = "song.wav"):
    """Synthesize music written in ABC notation into a playable WAV audio file."""
    import pathlib
    sandbox = get_active()
    # Ship our trusted ABC->WAV synthesizer into the sandbox and run it there on
    # the model's ABC (isolated, resource-limited), then copy the audio to host.
    synth_src = (pathlib.Path(__file__).resolve().parent / "music.py").read_text()
    if sandbox.run("cat > /workspace/_synth.py", stdin=synth_src).exit_code != 0:
        return "Error: could not stage synthesizer."
    if sandbox.run("cat > /workspace/_tune.abc", stdin=abc).exit_code != 0:
        return "Error: could not write ABC."
    result = sandbox.run("cd /workspace && python3 _synth.py _tune.abc _out.wav")
    if result.exit_code != 0:
        return result
    try:
        data = sandbox.read_bytes("/workspace/_out.wav")
    except RuntimeError as exc:
        return f"Error reading audio: {exc}"
    pathlib.Path(path).write_bytes(data)
    return f"{result.output.strip()}  ({len(data)} bytes written to host {path})"
