"""Correctness checks for the ABC->WAV synthesizer (no sandbox/GPU needed)."""
import wave

import numpy as np

from music import parse_abc, synthesize, abc_to_wav

# 1. Pitch mapping: A in C major is A4 = 440 Hz.
ev, _ = parse_abc("K:C\nA")
assert abs(ev[0]["freqs"][0] - 440.0) < 0.5, ev[0]["freqs"]
print("A4 ->", round(ev[0]["freqs"][0], 2), "Hz")

# 2. The synthesized tone really is 440 Hz (FFT of the actual audio).
samples = synthesize([{"freqs": [440.0], "seconds": 1.0}]).astype(np.float32)
spec = np.abs(np.fft.rfft(samples))
peak = np.fft.rfftfreq(len(samples), 1 / 44100)[np.argmax(spec)]
assert abs(peak - 440) < 5, peak
print("FFT dominant frequency ->", round(peak, 1), "Hz")

# 3. A C-major scale: 8 notes, lowercase c is an octave above uppercase C.
ev, _ = parse_abc("L:1/4\nK:C\nC D E F G A B c")
freqs = [e["freqs"][0] for e in ev]
assert len(ev) == 8, len(ev)
assert abs(freqs[0] - 261.63) < 1 and abs(freqs[-1] - 523.25) < 1, freqs
print("C-major scale ->", [round(f) for f in freqs])

# 4. Key signature: in G major, a bare F is sounded as F#.
f_in_c = parse_abc("K:C\nF")[0][0]["freqs"][0]
f_in_g = parse_abc("K:G\nF")[0][0]["freqs"][0]
assert f_in_g > f_in_c, (f_in_c, f_in_g)
print(f"F natural {f_in_c:.1f} Hz -> F# in G major {f_in_g:.1f} Hz")

# 5. Rests and durations: "C2 z2" -> a 2-unit note then a 2-unit rest.
ev, _ = parse_abc("L:1/4\nQ:1/4=120\nK:C\nC2 z2")
assert ev[0]["seconds"] == ev[1]["seconds"] and ev[1]["freqs"] == []
print(f"duration: C2 = {ev[0]['seconds']:.2f}s, rest z2 = {ev[1]['seconds']:.2f}s")

# 6. Render a recognizable tune to a real WAV.
meta, n = abc_to_wav(
    "X:1\nT:Twinkle\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
    "C C G G | A A G2 | F F E E | D D C2 |", "twinkle.wav")
w = wave.open("twinkle.wav"); dur = w.getnframes() / w.getframerate(); w.close()
print(f"twinkle.wav -> {n} notes, {dur:.1f}s, key {meta['key']}, {meta['bpm']:.0f} bpm")
assert dur > 3
print("\nall music synthesizer tests passed")
