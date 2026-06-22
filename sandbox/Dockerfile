# Minimal, throwaway environment for the Gemma 4 agent to act in.
# It holds NO secrets and gets no host mounts — the container's writable layer
# and an ephemeral tmpfs workspace are the only things the agent can touch.
FROM python:3.12-slim

# Inspection utilities + librsvg (in-sandbox SVG raster) + a C build toolchain
# and common library headers, so the agent can fetch and compile software (e.g.
# nginx) entirely inside the sandbox — it runs non-root on a read-only rootfs and
# cannot apt-install at runtime, so anything it needs to build must live here.
RUN apt-get update && apt-get install -y --no-install-recommends \
        procps file tree jq librsvg2-bin \
        curl wget ca-certificates tar xz-utils \
        gcc make libc6-dev \
        libpcre2-dev zlib1g-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Scientific Python so the agent's run_python tool is a real data-science brain:
# numeric (numpy), symbolic (sympy), tabular (pandas), plotting (matplotlib).
# Installed at build time; importable under the read-only root at runtime.
RUN pip install --no-cache-dir numpy sympy pandas matplotlib

# matplotlib + fontconfig need writable config/cache dirs; HOME and /var are
# read-only at runtime, so point both at the tmpfs (otherwise font rendering
# spews "No writable cache directories" into the agent's tool output).
# matplotlib also defaults to the headless Agg backend, so savefig() works.
ENV MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache

# Run as an unprivileged user; /workspace is the only place it writes.
RUN useradd --create-home --uid 1000 agent
WORKDIR /workspace

USER agent

# Stay alive so the harness can exec commands into the same container across
# multiple agent steps (preserving working state between tool calls).
CMD ["sleep", "infinity"]
