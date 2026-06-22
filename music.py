"""Synthesize a subset of ABC notation to a WAV file with numpy (no external synth).

Gemma 4 knows ABC notation well (it's a widely-used text music DSL, abundant in
training data as folk tunes), so it can *compose* in this language; we render the
text to audio. This is the SVG-studio idea for sound: model writes a domain DSL,
we turn it into a file.

Supported subset:
  Header fields  L: (unit note length, e.g. 1/8)   Q: (tempo, e.g. 1/4=120 or 120)
                 M: (meter, informational)          K: (key signature -> accidentals)
  Body           notes A-G / a-g, octave marks ' (up) and , (down),
                 accidentals ^ (sharp) _ (flat) = (natural), durations (C2, C/2,
                 C3/2), rests z, chords [CEG], bar lines | (ignored).

CLI:  python3 music.py tune.abc out.wav
"""

import re
import sys
import wave

import numpy as np

SR = 44100
_MAX_SECONDS = 180.0          # bound output length (untrusted input safety)
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Major key signatures -> letters that are sharp (+1) or flat (-1).
_SHARP_KEYS = {"G": "F", "D": "FC", "A": "FCG", "E": "FCGD", "B": "FCGDA", "F#": "FCGDAE"}
_FLAT_KEYS = {"F": "B", "Bb": "BE", "Eb": "BEA", "Ab": "BEAD", "Db": "BEADG"}

_TOKEN_RE = re.compile(
    r"\[(?P<chord>[^\]]*)\](?P<cdur>\d*/?\d*)"          # [CEG]2  chord + duration
    r"|(?P<acc>[\^_=]*)(?P<note>[A-Ga-gz])(?P<oct>[',]*)(?P<dur>\d*/?\d*)"
)
_CHORD_NOTE_RE = re.compile(r"(?P<acc>[\^_=]*)(?P<note>[A-Ga-g])(?P<oct>[',]*)")


def _key_accidentals(key):
    m = re.match(r"\s*([A-Ga-g])([#b]?)", key or "")
    if not m:
        return {}
    tonic = m.group(1).upper() + ("#" if m.group(2) == "#" else "b" if m.group(2) == "b" else "")
    if tonic in _SHARP_KEYS:
        return {n: 1 for n in _SHARP_KEYS[tonic]}
    if tonic in _FLAT_KEYS:
        return {n: -1 for n in _FLAT_KEYS[tonic]}
    return {}


def _parse_duration(s):
    if not s:
        return 1.0
    num, _, den = s.partition("/")
    n = int(num) if num else 1
    d = int(den) if den else (2 if "/" in s else 1)
    return n / d


def _note_freq(acc, letter, octmarks, key_acc):
    midi = 60 + _NATURAL[letter.upper()]      # uppercase C = middle C (midi 60)
    if letter.islower():
        midi += 12
    midi += 12 * octmarks.count("'") - 12 * octmarks.count(",")
    if acc:                                    # explicit accidental overrides key
        midi += acc.count("^") - acc.count("_")  # '=' (natural) contributes 0
    else:
        midi += key_acc.get(letter.upper(), 0)
    return 440.0 * 2 ** ((midi - 69) / 12)


def parse_abc(abc):
    """Parse ABC text -> (events, meta). Each event: {'freqs': [...], 'seconds': float}."""
    unit_len, beat_frac, bpm, key = 1 / 8, 1 / 4, 120.0, "C"
    body_lines, in_body = [], False
    for line in abc.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_body and re.match(r"^[A-Za-z]:", line):
            field, val = line[0], line[2:].strip()
            if field == "L":
                m = re.match(r"(\d+)/(\d+)", val)
                if m:
                    unit_len = int(m.group(1)) / int(m.group(2))
            elif field == "Q":
                mm = re.search(r"(\d+)/(\d+)\s*=\s*(\d+)", val)
                if mm:
                    beat_frac, bpm = int(mm.group(1)) / int(mm.group(2)), float(mm.group(3))
                elif re.search(r"\d+", val):
                    bpm = float(re.search(r"\d+", val).group())
            elif field == "K":
                key = val
                in_body = True            # K: ends the header
        elif in_body or not re.match(r"^[A-Za-z]:", line):
            in_body = True
            body_lines.append(line)

    key_acc = _key_accidentals(key)
    whole_note_seconds = (60.0 / bpm) / beat_frac
    body = " ".join(body_lines)

    events, total = [], 0.0
    for m in _TOKEN_RE.finditer(body):
        if m.group("chord") is not None:
            freqs = [_note_freq(c.group("acc"), c.group("note"), c.group("oct"), key_acc)
                     for c in _CHORD_NOTE_RE.finditer(m.group("chord"))]
            secs = _parse_duration(m.group("cdur")) * unit_len * whole_note_seconds
        else:
            note = m.group("note")
            secs = _parse_duration(m.group("dur")) * unit_len * whole_note_seconds
            freqs = [] if note == "z" else [
                _note_freq(m.group("acc"), note, m.group("oct"), key_acc)]
        if secs <= 0:
            continue
        events.append({"freqs": freqs, "seconds": secs})
        total += secs
        if total >= _MAX_SECONDS or len(events) > 4000:
            break
    return events, {"key": key, "bpm": bpm, "unit_len": unit_len, "seconds": total}


def _tone(freqs, seconds):
    n = max(1, int(SR * seconds))
    if not freqs:
        return np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SR
    wave_sum = np.zeros(n, dtype=np.float32)
    for f in freqs:                                   # additive: 3 harmonics
        wave_sum += np.sin(2 * np.pi * f * t)
        wave_sum += 0.5 * np.sin(2 * np.pi * 2 * f * t)
        wave_sum += 0.25 * np.sin(2 * np.pi * 3 * f * t)
    wave_sum /= len(freqs)
    # ADSR-ish envelope to avoid clicks and sound plucked.
    env = np.ones(n, dtype=np.float32)
    a = min(int(0.01 * SR), n // 2)        # 10 ms attack
    r = min(int(0.06 * SR), n // 2)        # 60 ms release
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] = np.linspace(1, 0, r)
    env *= np.linspace(1.0, 0.7, n)        # gentle decay
    return wave_sum * env


def synthesize(events):
    if not events:
        return np.zeros(1, dtype=np.int16)
    audio = np.concatenate([_tone(e["freqs"], e["seconds"]) for e in events])
    peak = np.max(np.abs(audio)) or 1.0
    return np.int16(audio / peak * 0.9 * 32767)


def abc_to_wav(abc, out_path):
    events, meta = parse_abc(abc)
    samples = synthesize(events)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return meta, len(events)


if __name__ == "__main__":
    abc = open(sys.argv[1]).read()
    out = sys.argv[2] if len(sys.argv) > 2 else "out.wav"
    meta, n = abc_to_wav(abc, out)
    print(f"synthesized {n} notes, {meta['seconds']:.1f}s, key {meta['key']}, "
          f"{meta['bpm']:.0f} bpm -> {out}")
