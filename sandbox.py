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
import os
import subprocess
from pathlib import Path

# Bump the tag whenever sandbox/Dockerfile changes so a stale image is rebuilt.
IMAGE_TAG = "gemma4-sandbox:v9"
_DOCKERFILE_DIR = Path(__file__).resolve().parent / "sandbox"

# Transparent MITM proxy: when networking is enabled the sandbox reaches the
# internet ONLY through this Squid container (built from source in
# sandbox/squid/), which re-signs every TLS leaf with our MITM CA. Bump the tag
# whenever anything under sandbox/squid/ changes.
PROXY_IMAGE_TAG = "gemma4-mitm:v3"
_PROXY_DOCKERFILE = _DOCKERFILE_DIR / "squid" / "Dockerfile"
_CA_CERT = _DOCKERFILE_DIR / "mitm" / "ca.crt"
_CA_KEY = _DOCKERFILE_DIR / "mitm" / "ca.key"

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
                 exec_timeout=60, exec_workspace=False, policy_file=None):
        self.container_id = None
        self.proxy_id = None
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.network = network
        self.exec_timeout = exec_timeout
        # Optional host YAML web-policy file, bind-mounted into the proxy and
        # hot-reloaded by the ICAP server. None => the baked allow-all default.
        self.policy_file = policy_file
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

    def _ensure_proxy_image(self):
        if not (_CA_CERT.exists() and _CA_KEY.exists()):
            raise RuntimeError(
                "MITM CA missing — run `make mitm-ca` first (generates "
                f"{_CA_CERT.parent}/ca.crt + ca.key; the key is gitignored)."
            )
        exists = subprocess.run(
            ["docker", "image", "inspect", PROXY_IMAGE_TAG],
            capture_output=True,
        ).returncode == 0
        if not exists:
            print(f"[sandbox] building MITM proxy image '{PROXY_IMAGE_TAG}' "
                  "from source (first run only; compiles Squid — slow)...")
            build = subprocess.run(
                # Context is sandbox/ so the Dockerfile can COPY mitm/ and squid/.
                ["docker", "build", "-t", PROXY_IMAGE_TAG,
                 "-f", str(_PROXY_DOCKERFILE), str(_DOCKERFILE_DIR)],
                capture_output=True, text=True,
            )
            if build.returncode != 0:
                raise RuntimeError(f"Proxy image build failed:\n{build.stderr}")

    def _start_proxy(self):
        """Start the privileged MITM proxy and wait for Squid to listen.

        The sandbox then joins this container's network namespace, so the
        proxy's iptables rules transparently intercept the agent's egress. Only
        the proxy is privileged (NET_ADMIN); the sandbox stays --cap-drop ALL.
        """
        self._ensure_proxy_image()
        args = [
            "docker", "run", "--detach", "--rm",
            # NET_ADMIN lets the entrypoint program iptables. We keep Docker's
            # default (already-restricted) cap set otherwise, because Squid
            # starts as root and needs SETUID/SETGID to drop to the squid user.
            # This is the ONLY privileged container; the sandbox stays cap-drop ALL.
            "--cap-add", "NET_ADMIN",
            # REDIRECT of locally-generated traffic lands on 127.0.0.1; the
            # kernel only routes that when route_localnet is enabled.
            "--sysctl", "net.ipv4.conf.all.route_localnet=1",
            # Make the shared netns IPv4-only: the interception is IPv4-only, so
            # a resolvable AAAA record would let the agent try IPv6 and bypass
            # it (then hit the ip6tables DROP and just hang). With no IPv6 addr,
            # getaddrinfo (AI_ADDRCONFIG) returns only A records.
            "--sysctl", "net.ipv6.conf.all.disable_ipv6=1",
            "--sysctl", "net.ipv6.conf.default.disable_ipv6=1",
            "--memory", self.memory, "--cpus", self.cpus,
        ]
        # Override the baked allow-all policy with the operator's file, mounted
        # read-only over the same path so the ICAP server hot-reloads it. Must be
        # world-readable (the ICAP server runs as the unprivileged squid uid).
        if self.policy_file:
            host_policy = os.path.abspath(self.policy_file)
            if not os.path.isfile(host_policy):
                raise RuntimeError(f"policy file not found: {host_policy}")
            args += ["-v", f"{host_policy}:/etc/squid/policy.yaml:ro"]
        args.append(PROXY_IMAGE_TAG)
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start MITM proxy:\n{result.stderr}")
        self.proxy_id = result.stdout.strip()
        atexit.register(self._stop_proxy)

        # Wait for both the ICAP policy server (:1344) and Squid's intercept
        # port (:3130) to listen. The entrypoint installs the iptables rules and
        # starts ICAP BEFORE exec'ing Squid, so a listening :3130 implies the
        # redirect is in place and ICAP came up.
        import time
        for _ in range(60):
            probe = subprocess.run(
                ["docker", "exec", self.proxy_id, "sh", "-c",
                 "ss -ltn 2>/dev/null | grep -q ':3130' && "
                 "ss -ltn 2>/dev/null | grep -q ':1344'"],
                capture_output=True,
            )
            if probe.returncode == 0:
                pol = self.policy_file or "default allow-all"
                print(f"[sandbox] MITM proxy {self.proxy_id[:12]} ready "
                      f"(intercepting :80/:443, egress fail-closed, policy={pol})")
                return
            time.sleep(0.5)
        logs = subprocess.run(["docker", "logs", self.proxy_id],
                              capture_output=True, text=True)
        self._stop_proxy()
        raise RuntimeError(f"MITM proxy did not become ready:\n{logs.stderr}\n{logs.stdout}")

    def _stop_proxy(self):
        if self.proxy_id:
            subprocess.run(["docker", "kill", self.proxy_id], capture_output=True)
            print(f"[sandbox] stopped MITM proxy {self.proxy_id[:12]}")
            self.proxy_id = None

    def start(self):
        global _ACTIVE
        self._ensure_image()
        # Networking is always MITM'd: bring up the proxy first, then join its
        # network namespace so all of the sandbox's egress is intercepted.
        if self.network:
            self._start_proxy()
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
        if self.network:
            # Share the proxy's netns; its iptables rules intercept our traffic.
            args += ["--network", f"container:{self.proxy_id}"]
        else:
            args += ["--network", "none"]
        args.append(IMAGE_TAG)

        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            # Don't leave the proxy running if the sandbox failed to come up.
            self._stop_proxy()
            raise RuntimeError(f"Failed to start sandbox:\n{result.stderr}")
        self.container_id = result.stdout.strip()
        _ACTIVE = self
        # Defensive: tear the container down even if the process crashes.
        atexit.register(self.stop)
        net = "mitm" if self.network else "off"
        print(f"[sandbox] started container {self.container_id[:12]} (network={net})")
        return self

    def stop(self):
        global _ACTIVE
        # Kill the sandbox BEFORE the proxy: the sandbox joins the proxy's
        # network namespace, so the netns owner must outlive its joiner.
        if self.container_id:
            subprocess.run(
                ["docker", "kill", self.container_id],
                capture_output=True,
            )
            print(f"[sandbox] stopped container {self.container_id[:12]}")
            self.container_id = None
        self._stop_proxy()
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

    def export_workspace(self, host_dir):
        """Copy everything under /workspace out to a host directory.

        Same constraint as read_bytes: /workspace is a tmpfs mount that
        `docker cp` can't read, but an exec'd process inside the namespace can.
        So we tar the tree to stdout (`tar c -C /workspace .`), base64 it across
        the exec boundary, and untar it on the host. An empty workspace yields an
        empty (but created) host_dir rather than an error — a run that wrote no
        files is a valid, gradeable outcome.

        Returns the host Path written to.
        """
        if self.container_id is None:
            raise RuntimeError("Sandbox is not running.")
        host_dir = Path(host_dir)
        host_dir.mkdir(parents=True, exist_ok=True)
        # tar's exit code is non-zero if /workspace vanished; an empty dir tars fine.
        r = subprocess.run(
            ["docker", "exec", "--user", "1000:1000", self.container_id,
             "sh", "-c", "tar c -C /workspace . | base64"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"export_workspace failed: {r.stderr.strip()}")
        import base64
        tar_bytes = base64.b64decode(r.stdout)
        if not tar_bytes:
            return host_dir
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
            tf.extractall(host_dir)
        return host_dir
