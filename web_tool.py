"""browse — gives the agent READING on the web.

Registered only when the harness runs with --network. The fetch happens INSIDE
the sandbox (webfetch.py is staged in and run there, exactly as compose_music
stages music.py), which is the whole point: the request then travels the same
intercepted path as any other sandbox egress, so the operator's web policy
(block / redirect / set-header / rewrite-body) governs what the agent can read,
and every request lands in the proxy's network audit.

Nothing here needs proxy configuration — the sandbox's iptables rules redirect
:80/:443 to Squid and its trust store holds only the MITM CA.
"""

import pathlib
import shlex

from sandbox import get_active
from tools import tool

_SCRIPT = pathlib.Path(__file__).resolve().parent / "webfetch.py"


@tool
def browse(url: str, question: str = ""):
    """Fetch a web page over the sandbox's monitored network and return its readable text. Optionally focus the excerpt on a question."""
    sandbox = get_active()
    try:
        src = _SCRIPT.read_text()
    except OSError as exc:
        return f"Error: could not read the fetcher script ({exc})."
    if sandbox.run("cat > /workspace/_webfetch.py", stdin=src).exit_code != 0:
        return "Error: could not stage the web fetcher into the sandbox."
    cmd = f"python3 /workspace/_webfetch.py {shlex.quote(url)}"
    if question:
        cmd += f" {shlex.quote(question)}"
    result = sandbox.run(cmd)
    # A non-zero exit is an environment fact worth showing verbatim (the policy
    # may have blocked the request); dispatch() stringifies ToolResult with it.
    return result if result.exit_code != 0 else result.output.strip()
