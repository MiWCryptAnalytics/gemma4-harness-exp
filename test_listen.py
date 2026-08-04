"""The hearing path: WAV -> analysis -> picture -> vision engine.

Signal analysis (load_wav / spectrogram / facts) is pure numpy and runs
everywhere. Rendering needs matplotlib, which lives in the sandbox image rather
than the host venv, so the render + end-to-end listen() checks are gated on
matplotlib and Docker respectively and skip cleanly without them.
"""
import base64
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

import audioviz
from audioviz import facts, load_wav, spectrogram
from music import abc_to_wav

TUNE = ("X:1\nT:Test\nM:4/4\nL:1/4\nQ:1/4=180\nK:C\n"
        "C D E F | G A B c |")

tmpdir = tempfile.mkdtemp()
wav_path = os.path.join(tmpdir, "scale.wav")
abc_to_wav(TUNE, wav_path, engine="sine")

# --- load_wav: decoded, mixed to mono, bounded ----------------------------
samples, rate, channels, duration = load_wav(wav_path)
with wave.open(wav_path) as w:
    expect_rate, expect_ch = w.getframerate(), w.getnchannels()
assert rate == expect_rate and channels == expect_ch, (rate, channels)
assert samples.ndim == 1, "mixed down to mono"
# music.py normalizes to a 0.9 peak; decoding must preserve that scale.
assert 0.85 < np.abs(samples).max() <= 1.0, np.abs(samples).max()
assert 2.0 < duration < 4.0, duration
print(f"load_wav -> {len(samples)} samples, {rate} Hz, {channels}ch, {duration:.2f}s")

# The length bound applies to untrusted input (a very long WAV can't be loaded whole).
short, _r, _c, full_duration = load_wav(wav_path, max_seconds=0.5)
assert len(short) <= rate * 0.5 + 1, len(short)
assert abs(full_duration - duration) < 0.01, "duration reports the FILE, not the slice"
print(f"max_seconds bound: {len(short)} samples loaded, duration still {full_duration:.2f}s")

# --- spectrogram: real pitch content, in dB -------------------------------
spec_db, top_hz = spectrogram(samples, rate)
assert spec_db.ndim == 2 and spec_db.shape[1] > 1, spec_db.shape
# The frequency axis adapts to where the energy is, within the fixed bounds.
bin_hz = rate / audioviz.FFT_SIZE
assert audioviz.MIN_DISPLAY_HZ - bin_hz <= top_hz <= audioviz.MAX_DISPLAY_HZ + bin_hz, top_hz
# The scale's first note is middle C (261.6 Hz); its bin must be loud early on.
freqs = np.fft.rfftfreq(audioviz.FFT_SIZE, 1.0 / rate)[: spec_db.shape[0]]
first_col = spec_db[:, 2]
peak_hz = freqs[int(np.argmax(first_col))]
assert abs(peak_hz - 261.63) < 30, f"expected ~C4 at onset, got {peak_hz:.0f} Hz"
print(f"spectrogram -> {spec_db.shape[0]}x{spec_db.shape[1]} bins to {top_hz:.0f} Hz, "
      f"onset peak {peak_hz:.0f} Hz (C4)")

# A tiny file shorter than one FFT window must still render, not crash.
tiny, _r, _c, _d = load_wav(wav_path, max_seconds=0.001)
assert spectrogram(tiny if len(tiny) else np.zeros(4), rate)[0].shape[1] >= 1
print("sub-window audio still produces a spectrogram: OK")

# --- facts: the measured line the model is given alongside the picture ----
line = facts(samples, rate, channels, duration)
for token in ("duration", "Hz", "dBFS"):
    assert token in line, line
assert facts(np.zeros(10), rate, 1, 0.0).count("-120.0 dBFS") == 2, "silence -> floor"
print("facts ->", line)

# --- render: needs matplotlib (sandbox image), skipped on the host --------
try:
    import matplotlib  # noqa: F401
    have_mpl = True
except ImportError:
    have_mpl = False

if have_mpl:
    png = os.path.join(tmpdir, "scale.png")
    summary = audioviz.render(wav_path, png)
    assert os.path.getsize(png) > 5000, os.path.getsize(png)
    assert "duration" in summary, summary
    print(f"render -> {os.path.getsize(png)} byte PNG, summary {summary!r}")
else:
    print("matplotlib not on the host (it lives in the sandbox image) -> "
          "render test skipped; the Docker path below covers it")


# --- end-to-end listen() through a real sandbox --------------------------
def _docker_available():
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


class FakeVision:
    """Asserts the tool really decoded a picture before asking the model."""

    def __init__(self):
        self.seen = None

    def ask_image(self, image, question, max_new_tokens=320):
        assert image.width > 200 and image.height > 100, image.size
        self.seen = question
        return f"[fake-vision {image.size}] sounds like a rising scale."


if not _docker_available():
    print("docker unavailable: skipping the end-to-end listen() check")
else:
    import audio_tool
    from sandbox import Sandbox

    fake = FakeVision()
    audio_tool.set_engine(fake)
    with Sandbox() as sb:
        b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
        assert sb.run("base64 -d > /workspace/scale.wav", stdin=b64).exit_code == 0

        out = audio_tool.listen("scale.wav", "Is the melody rising or falling?")
        print("listen ->", out.replace("\n", " | ")[:160])
        assert "fake-vision" in out, out
        assert "duration" in out and "Hz" in out, "measured facts must be returned"
        assert "rising or falling" in (fake.seen or ""), fake.seen
        assert "spectrogram" in (fake.seen or ""), "prompt must explain the picture"

        missing = audio_tool.listen("no_such_file.wav")
        assert missing.startswith("Error"), missing
        print("missing file ->", missing.splitlines()[0][:80])

    # Without an engine the tool refuses instead of raising.
    audio_tool.set_engine(None)
    assert audio_tool.listen("scale.wav").startswith("Error: hearing needs")
    print("no vision engine -> clean refusal: OK")

shutil.rmtree(tmpdir, ignore_errors=True)
print("\nall listen tests passed")
