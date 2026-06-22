"""Docker-backed execution sandbox for the Gemma 4 agent.

The model runs on the host (it needs the GPU), but every action it takes runs
inside a locked-down container so it cannot touch the host. Isolation applied:

  * no host bind mounts          -> host filesystem is invisible
  * --network none               -> no network access from tools
  * --read-only root filesystem  -> base image can't be modified
  * tmpfs /workspace + /tmp      -> writes are ephemeral, vanish on stop
  * non-root user (uid 1000)     -> no privileged actions
  * --cap-drop ALL + no-new-privileges
  * --pids/--memory/--cpus limits -> bounds a runaway or fork-bomb

A single container stays alive for the whole session so state (files, cwd)
persists across the agent's tool calls; it is removed on exit.
"""

import atexit
import subprocess
from pathlib import Path

# Bump the tag whenever sandbox/Dockerfile changes so a stale image is rebuilt.
IMAGE_TAG = "gemma4-sandbox:v7"
_DOCKERFILE_DIR = Path(__file__).resolve().parent / "sandbox"

# Set by Sandbox.start(); the env tools in tools.py route through this.
_ACTIVE = None


def get_active():
    """Return the running sandbox, or raise if the harness forgot to start one."""
    if _ACTIVE is None:
        raise RuntimeError("No active sandbox. Start one with `with Sandbox():`.")
    return _ACTIVE


class ToolResult:
    """Outcome of a command run in the sandbox."""

    def __init__(self, exit_code, output):
        self.exit_code = exit_code
        self.output = output

    def __str__(self):
        body = self.output.strip()
        return f"(exit {self.exit_code})\n{body}" if body else f"(exit {self.exit_code})"


class Sandbox:
    """Lifecycle manager for the isolated execution container."""

    def __init__(self, memory="2g", cpus="2", pids_limit=256, network=False,
                 exec_timeout=60, exec_workspace=False):
        self.container_id = None
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network = network
        self.exec_timeout = exec_timeout
        # The tmpfs workspace is noexec by default (a dropped binary can't run).
        # Building software (./configure, compiled output) needs exec, so it's an
        # explicit opt-in rather than the default.
        self.exec_workspace = exec_workspace

    # -- lifecycle ---------------------------------------------------------

    def _ensure_image(self):
        exists = subprocess.run(
            ["docker", "image", "inspect", IMAGE_TAG],
            capture_output=True,
        ).returncode == 0
        if not exists:
            print(f"[sandbox] building image '{IMAGE_TAG}' (first run only)...")
            build = subprocess.run(
                ["docker", "build", "-t", IMAGE_TAG, str(_DOCKERFILE_DIR)],
                capture_output=True, text=True,
            )
            if build.returncode != 0:
                raise RuntimeError(f"Image build failed:\n{build.stderr}")

    def start(self):
        global _ACTIVE
        self._ensure_image()
        ex = "exec," if self.exec_workspace else ""
        args = [
            "docker", "run", "--detach", "--rm",
            "--read-only",
            "--tmpfs", f"/workspace:{ex}uid=1000,mode=1777",
            "--tmpfs", f"/tmp:{ex}mode=1777",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(self.pids_limit),
            "--memory", self.memory,
            "--cpus", str(self.cpus),
            "--workdir", "/workspace",
        ]
        if not self.network:
            args += ["--network", "none"]
        args.append(IMAGE_TAG)

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start sandbox:\n{result.stderr}")
        self.container_id = result.stdout.strip()
        _ACTIVE = self
        # Defensive: tear the container down even if the process crashes.
        atexit.register(self.stop)
        print(f"[sandbox] started container {self.container_id[:12]} (network={'on' if self.network else 'off'})")
        return self

    def stop(self):
        global _ACTIVE
        if self.container_id:
            subprocess.run(
                ["docker", "kill", self.container_id],
                capture_output=True,
            )
            print(f"[sandbox] stopped container {self.container_id[:12]}")
            self.container_id = None
        if _ACTIVE is self:
            _ACTIVE = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- execution ---------------------------------------------------------

    def run(self, command, stdin=None):
        """Run a shell command inside the container as the agent user."""
        if self.container_id is None:
            raise RuntimeError("Sandbox is not running.")
        exec_args = ["docker", "exec", "--user", "1000:1000", "--workdir", "/workspace"]
        if stdin is not None:
            exec_args.append("--interactive")
        exec_args += [self.container_id, "sh", "-c", command]
        import time
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                exec_args,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.exec_timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(124, f"Timed out after {self.exec_timeout}s.")
        from instrument import debug
        debug(f"sandbox$ {command[:50]!r} -> exit {proc.returncode} "
              f"in {time.perf_counter() - t0:.2f}s")
        return ToolResult(proc.returncode, (proc.stdout or "") + (proc.stderr or ""))

    def read_bytes(self, container_path):
        """Read a file's raw bytes out of the sandbox (for binary results, e.g. PNG).

        Uses `base64` over `docker exec` rather than `docker cp`: our writable area
        is a tmpfs mount, and `docker cp` can't read tmpfs, but an exec'd process
        runs inside the namespace and sees it.
        """
        if self.container_id is None:
            raise RuntimeError("Sandbox is not running.")
        r = subprocess.run(
            ["docker", "exec", "--user", "1000:1000", self.container_id,
             "base64", container_path],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"read_bytes failed: {r.stderr.strip()}")
        import base64
        return base64.b64decode(r.stdout)
