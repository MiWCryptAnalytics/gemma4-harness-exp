"""Render a WAV file as a picture the vision tower can read.

Staged into the sandbox and run there (like music.py for compose_music), because
the sandbox is where the audio lives and where numpy/matplotlib are installed.
Produces one PNG with two stacked panels:

    top     waveform envelope — dynamics, phrasing, silence
    bottom  log-magnitude spectrogram — pitch contour, harmonics, rhythm

and prints a one-line factual summary to stdout (duration, rate, channels, peak)
so the harness can report measured facts alongside the model's impression.

    python3 audioviz.py in.wav out.png
"""

import sys
import wave

MAX_SECONDS = 300.0     # bound the render for untrusted input (cf. music.py)
FFT_SIZE = 1024
HOP = 256
MAX_DISPLAY_HZ = 8000.0
MIN_DISPLAY_HZ = 1500.0     # never zoom in past this, so context stays visible


def load_wav(path, max_seconds=MAX_SECONDS):
    """(mono float array in [-1,1], sample_rate, channels, duration_seconds)."""
    import numpy as np

    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        n_frames = w.getnframes()
        duration = n_frames / rate if rate else 0.0
        keep = min(n_frames, int(max_seconds * rate)) if rate else 0
        raw = w.readframes(keep)

    if width == 1:                       # 8-bit WAV is unsigned
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {width * 8}-bit")

    if channels > 1:
        usable = (len(data) // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return data, rate, channels, duration


def spectrogram(samples, rate, fft_size=FFT_SIZE, hop=HOP):
    """(log-magnitude matrix [freq, time], max displayed frequency).

    The frequency axis is trimmed to where the energy actually is, so a melody
    whose fundamentals sit under 1 kHz fills the panel instead of hugging the
    bottom of an 8 kHz plot — the contour is what the vision tower has to read.
    """
    import numpy as np

    if len(samples) < fft_size:
        samples = np.pad(samples, (0, fft_size - len(samples)))
    window = np.hanning(fft_size)
    n_frames = 1 + (len(samples) - fft_size) // hop
    # Cap the frame count so a long file can't blow up memory in the sandbox.
    n_frames = max(1, min(n_frames, 4000))
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, fft_size),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    ) * window
    spec = np.abs(np.fft.rfft(frames, axis=1)).T
    freqs = np.fft.rfftfreq(fft_size, 1.0 / rate)

    # Keep everything up to the bin holding 99% of the total energy, clamped so
    # the view is neither uselessly zoomed nor mostly empty noise floor.
    energy = spec.sum(axis=1)
    total = energy.sum()
    if total > 0:
        idx = int(np.searchsorted(np.cumsum(energy), 0.99 * total))
        limit_hz = float(freqs[min(idx, len(freqs) - 1)]) * 1.3
    else:
        limit_hz = MIN_DISPLAY_HZ
    limit_hz = min(max(limit_hz, MIN_DISPLAY_HZ), MAX_DISPLAY_HZ, float(freqs[-1]))

    cutoff = int(np.searchsorted(freqs, limit_hz)) + 1
    cutoff = min(cutoff, len(freqs))
    spec = spec[:cutoff]
    return 20.0 * np.log10(spec + 1e-6), float(freqs[cutoff - 1])


def facts(samples, rate, channels, duration):
    """One human/model-readable line of measured facts about the audio."""
    import numpy as np

    peak = float(np.abs(samples).max()) if len(samples) else 0.0
    rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) else 0.0
    peak_db = 20.0 * np.log10(peak) if peak > 0 else -120.0
    rms_db = 20.0 * np.log10(rms) if rms > 0 else -120.0
    kind = {1: "mono", 2: "stereo"}.get(channels, f"{channels}ch")
    return (f"duration {duration:.1f}s, {rate} Hz, {kind}, "
            f"peak {peak_db:.1f} dBFS, rms {rms_db:.1f} dBFS")


def render(wav_path, png_path):
    """Write the two-panel PNG; return the facts line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    samples, rate, channels, duration = load_wav(wav_path)
    if not len(samples):
        raise ValueError("audio file contains no samples")
    summary = facts(samples, rate, channels, duration)

    spec_db, top_hz = spectrogram(samples, rate)
    shown = min(duration, len(samples) / rate)

    fig, (ax_wave, ax_spec) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [1, 2]})

    t = np.linspace(0, shown, len(samples))
    ax_wave.plot(t, samples, linewidth=0.4, color="#1f77b4")
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_ylabel("amplitude")
    ax_wave.set_title(f"{wav_path} — {summary}", fontsize=9)
    ax_wave.grid(alpha=0.25)

    ax_spec.imshow(spec_db, origin="lower", aspect="auto",
                   extent=[0, shown, 0, top_hz], cmap="magma")
    ax_spec.set_ylabel("frequency (Hz)")
    ax_spec.set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: audioviz.py in.wav out.png", file=sys.stderr)
        raise SystemExit(2)
    print(render(sys.argv[1], sys.argv[2]))
