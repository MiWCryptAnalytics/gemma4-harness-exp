"""listen — gives the agent EARS.

Gemma 4 has no audio encoder here, so hearing is routed through the sense it does
have: audioviz.py renders the WAV as a waveform + spectrogram picture inside the
sandbox, and the model's own vision tower reads it. Measured facts (duration,
sample rate, peak level) come back alongside, so the model isn't guessing at
numbers it can only eyeball.

Together with compose_music this closes a loop the agent can drive on its own:
compose -> listen -> revise.

Registering it (importing this module) adds `listen`; call set_engine() with the
agent's vision-capable UnifiedEngine first, as vision_tool does.
"""

import pathlib
import shlex
from io import BytesIO

from sandbox import get_active
from tools import tool

_ENGINE = None
_SCRIPT = pathlib.Path(__file__).resolve().parent / "audioviz.py"


def set_engine(engine):
    """Point the hearing tool at the agent's (vision-capable) engine."""
    global _ENGINE
    _ENGINE = engine


@tool
def listen(path: str, question: str = "Describe this audio: its character, rhythm, and pitch range."):
    """Listen to a WAV audio file in the sandbox and answer a question about how it sounds."""
    if _ENGINE is None:
        return "Error: hearing needs the vision engine (run with --vision)."
    sandbox = get_active()
    container_path = path if path.startswith("/") else f"/workspace/{path}"

    try:
        src = _SCRIPT.read_text()
    except OSError as exc:
        return f"Error: could not read the audio renderer ({exc})."
    if sandbox.run("cat > /workspace/_audioviz.py", stdin=src).exit_code != 0:
        return "Error: could not stage the audio renderer into the sandbox."

    png = "/workspace/_listen.png"
    result = sandbox.run(f"python3 /workspace/_audioviz.py "
                         f"{shlex.quote(container_path)} {shlex.quote(png)}")
    if result.exit_code != 0:
        return f"Error rendering audio: {result}"
    measured = result.output.strip().splitlines()[-1] if result.output.strip() else ""

    try:
        data = sandbox.read_bytes(png)
    except RuntimeError as exc:
        return f"Error reading the rendered audio image: {exc}"
    try:
        from PIL import Image
        img = Image.open(BytesIO(data))
        img.load()
    except Exception as exc:
        return f"Error: rendered audio is not a readable image ({type(exc).__name__}: {exc})."

    prompt = (
        "This image shows an audio recording: the waveform envelope on top and a "
        "log-frequency spectrogram below with octave Cs marked on the axis "
        f"(measured: {measured}). How to read instrumental audio here: one "
        "pitched note appears as a STACK of parallel horizontal harmonic bands — "
        "that is the normal signature of a single ordinary instrument, not "
        "complexity or dissonance. The melody is the LOWEST bright band; follow "
        "it against the C gridlines to get the pitch contour. In the waveform, a "
        "sharp attack followed by a decay is the normal envelope of a piano or "
        "plucked note, not aggression — count attacks for the rhythm. With that "
        f"in mind, answer: {question}"
    )
    return f"{measured}\n{_ENGINE.ask_image(img, prompt)}"
