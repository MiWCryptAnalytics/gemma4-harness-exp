"""Correctness checks for the ABC->WAV synthesizer (no sandbox/GPU needed)."""
import wave

import numpy as np

import music
from music import parse_abc, synthesize, abc_to_wav


def events(abc):
    """First voice's events — most checks use single-voice tunes."""
    voices, _ = parse_abc(abc)
    return voices[0]["events"]


# 1. Pitch mapping: A in C major is A4 = 440 Hz.
ev = events("K:C\nA")
assert abs(ev[0]["freqs"][0] - 440.0) < 0.5, ev[0]["freqs"]
print("A4 ->", round(ev[0]["freqs"][0], 2), "Hz")

# 2. The synthesized tone really is 440 Hz (FFT of the actual audio).
samples = synthesize([{"freqs": [440.0], "seconds": 1.0}]).astype(np.float32)
spec = np.abs(np.fft.rfft(samples))
peak = np.fft.rfftfreq(len(samples), 1 / 44100)[np.argmax(spec)]
assert abs(peak - 440) < 5, peak
print("FFT dominant frequency ->", round(peak, 1), "Hz")

# 3. A C-major scale: 8 notes, lowercase c is an octave above uppercase C.
ev = events("L:1/4\nK:C\nC D E F G A B c")
freqs = [e["freqs"][0] for e in ev]
assert len(ev) == 8, len(ev)
assert abs(freqs[0] - 261.63) < 1 and abs(freqs[-1] - 523.25) < 1, freqs
print("C-major scale ->", [round(f) for f in freqs])

# 4. Key signature: in G major, a bare F is sounded as F#.
f_in_c = events("K:C\nF")[0]["freqs"][0]
f_in_g = events("K:G\nF")[0]["freqs"][0]
assert f_in_g > f_in_c, (f_in_c, f_in_g)
print(f"F natural {f_in_c:.1f} Hz -> F# in G major {f_in_g:.1f} Hz")

# 5. Rests and durations: "C2 z2" -> a 2-unit note then a 2-unit rest.
ev = events("L:1/4\nQ:1/4=120\nK:C\nC2 z2")
assert ev[0]["seconds"] == ev[1]["seconds"] and ev[1]["freqs"] == []
print(f"duration: C2 = {ev[0]['seconds']:.2f}s, rest z2 = {ev[1]['seconds']:.2f}s")

TWINKLE = ("X:1\nT:Twinkle\nM:4/4\nL:1/4\nQ:1/4=120\nK:C\n"
           "C C G G | A A G2 | F F E E | D D C2 |")

# 6. Render a recognizable tune to a real WAV (sine engine: sandbox-safe path).
meta, n = abc_to_wav(TWINKLE, "twinkle.wav", engine="sine")
w = wave.open("twinkle.wav"); dur = w.getnframes() / w.getframerate(); w.close()
assert meta["engine"] == "sine" and dur > 3
print(f"twinkle.wav -> {n} notes, {dur:.1f}s, key {meta['key']}, {meta['bpm']:.0f} bpm "
      f"[{meta['engine']}]")

# 7. Notes carry MIDI numbers for the fluidsynth engine (A4 = midi 69).
voices, meta = parse_abc("%%MIDI program 40\nK:C\nA")
assert voices[0]["events"][0]["midi"] == [69], voices[0]["events"][0]
assert meta["program"] == 40 and voices[0]["program"] == 40, meta
print("A4 -> midi", voices[0]["events"][0]["midi"][0],
      "| %%MIDI program ->", meta["program"])

# 8. Soundfont rendering: the same tune through fluidsynth, when pyfluidsynth
# and the GeneralUser GS soundfont are installed.
HAVE_FLUID = music.fluidsynth is not None and music.find_soundfont()
if HAVE_FLUID:
    meta, n = abc_to_wav(TWINKLE, "twinkle_sf.wav", engine="fluidsynth")
    w = wave.open("twinkle_sf.wav")
    ch, dur = w.getnchannels(), w.getnframes() / w.getframerate()
    frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    w.close()
    assert n == 14 and ch == 2 and dur > 8 and np.abs(frames).max() > 1000
    print(f"twinkle_sf.wav -> {n} notes, stereo {dur:.1f}s via {meta['soundfont']}")
else:
    print("fluidsynth unavailable -> soundfont test skipped (sine fallback covers it)")

# 9. Multi-voice, multi-instrument ABC (a real Gemma 4 generation): three named
# voices must play IN PARALLEL on distinct instruments, not one after another.
NEON = """X:1
T:Neon Overdrive
C:AI Assistant
M:4/4
L:1/8
Q:1/4=140
K:Am
%%MIDI program 0
V:1 name="Lead_Guitar"
|: A2 c2 e2 a2 | g2 e2 c2 A2 | f2 d2 B2 G2 | E4 z2 A2 :|
|: c2 e2 a2 c2 | b2 g2 e2 d2 | e2 f2 g2 a2 | g4 z2 e2 :|
V:2 name="Bass_Guitar"
|: A,2 A,2 A,2 A,2 | G,,2 G,,2 G,,2 G,,2 | F,,2 F,,2 F,,2 F,,2 | E,,4 z2 A,,2 :|
|: C,2 C,2 C,2 C,2 | B,,2 B,,2 B,,2 B,,2 | E,,2 E,,2 E,,2 E,,2 | G,,4 z2 E,,2 :|
V:3 name="Drums"
|: [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] :|
|: [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] | [F2F2] [F2F2] [F2F2] [F2F2] :|
"""
voices, meta = parse_abc(NEON)
assert [v["name"] for v in voices] == ["Lead_Guitar", "Bass_Guitar", "Drums"], voices
assert voices[0]["program"] == 29, voices[0]        # guitar, from the voice name
assert voices[1]["program"] == 33, voices[1]        # bass wins over guitar
assert voices[2]["drums"], voices[2]                # -> GM percussion channel
# The V: header lines must not leak note letters ("name=..." contains a, e, d).
assert len(voices[0]["events"]) == 30, len(voices[0]["events"])
# All voices span the same wall-clock time: they sound together, not in a row.
spans = [v["seconds"] for v in voices]
assert max(spans) - min(spans) < 0.01, spans
print(f"multi-voice parse -> {[(v['name'], v['program']) for v in voices]}, "
      f"each {spans[0]:.1f}s")

if HAVE_FLUID:
    meta, n = abc_to_wav(NEON, "neon.wav", engine="fluidsynth")
    w = wave.open("neon.wav")
    dur = w.getnframes() / w.getframerate()
    frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    w.close()
    # Parallel voices: total length is ONE voice span (+0.5s ring-out), nowhere
    # near the ~41s the three voices would take played back-to-back.
    assert abs(dur - (spans[0] + 0.5)) < 0.05, dur
    assert np.abs(frames).max() > 1000
    print(f"neon.wav -> {n} notes in 3 instruments, {dur:.1f}s (parallel, not "
          f"{sum(spans):.0f}s sequential)")
else:
    print("fluidsynth unavailable -> multi-instrument render skipped")

print("\nall music synthesizer tests passed")
