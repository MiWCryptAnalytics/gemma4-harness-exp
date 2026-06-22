"""Preflight check — run `make doctor` (or `python doctor.py`) before first use.

Verifies the environment is ready and prints an actionable hint for anything
missing: Python dependencies, the Docker daemon, a GPU, and the model weights.
Exit code is non-zero only when a HARD requirement (deps or Docker) is missing;
a GPU and the model are warnings (the no-GPU `make test` / `make dry-run` paths
work without them).
"""

import importlib
import os
import shutil
import subprocess
import sys

MODEL_ID = os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-12b-it")

OK, WARN, BAD = "\033[32m✓\033[0m", "\033[33m⚠\033[0m", "\033[31m✗\033[0m"


def line(sym, label, detail=""):
    print(f"  {sym} {label}" + (f" — {detail}" if detail else ""))


def check_deps():
    mods = {"torch": "torch", "transformers": "transformers",
            "accelerate": "accelerate", "torchvision": "torchvision",
            "PIL": "pillow", "numpy": "numpy"}
    missing = []
    for mod, pip in mods.items():
        try:
            m = importlib.import_module(mod)
            line(OK, pip, getattr(m, "__version__", "?"))
        except Exception:
            missing.append(pip)
            line(BAD, pip, "missing")
    if missing:
        line(" ", "", f"fix: ./venv/bin/pip install {' '.join(missing)}")
    return not missing


def check_docker():
    if not shutil.which("docker"):
        line(BAD, "docker", "not installed — see https://docs.docker.com/get-docker/")
        return False
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        line(BAD, "docker daemon", "not running — start Docker and retry")
        return False
    line(OK, "docker", "daemon running")
    return True


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            line(OK, "gpu", f"{name} ({gb:.0f} GB)")
            if gb < 22:
                line(" ", "", "the 12B model wants ~24 GB; expect CPU offload")
            return True
        line(WARN, "gpu", "no CUDA GPU — real model runs need one")
        return False
    except Exception as exc:
        line(WARN, "gpu", f"could not check ({exc})")
        return False


def check_model():
    hub = os.path.join(os.environ.get("HF_HOME")
                       or os.path.expanduser("~/.cache/huggingface"), "hub")
    folder = "models--" + MODEL_ID.replace("/", "--")
    if os.path.isdir(os.path.join(hub, folder)):
        line(OK, "model", f"{MODEL_ID} in cache")
        return True
    line(WARN, "model", f"{MODEL_ID} not found in HF cache")
    line(" ", "", "accept the Gemma terms, then `./venv/bin/huggingface-cli login` "
                  "(downloads on first run)")
    return False


def main():
    print("Gemma 4 harness — preflight\n")
    print(" Python dependencies:")
    deps = check_deps()
    print("\n Runtime:")
    docker = check_docker()
    gpu = check_gpu()
    model = check_model()

    print()
    if deps and docker:
        extra = "" if (gpu and model) else "  (warnings above only affect real model runs)"
        print(f"\033[32mReady.\033[0m{extra}")
        sys.exit(0)
    print("\033[31mNot ready\033[0m — fix the ✗ items above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
