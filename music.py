"""Synthesize a subset of ABC notation to a WAV file.

Gemma 4 knows ABC notation well (it's a widely-used text music DSL, abundant in
training data as folk tunes), so it can *compose* in this language; we render the
text to audio. This is the SVG-studio idea for sound: model writes a domain DSL,
we turn it into a file.

Two rendering engines:
  fluidsynth  Real instrument samples via pyfluidsynth + a General MIDI
              soundfont (auto-discovered, see find_soundfont). Used when
              available. Voices play on separate MIDI channels with their own
              instruments; a voice named like "Drums" plays GM percussion.
  sine        Additive sine synthesis with numpy, no external dependencies.
              Fallback used inside the sandbox when libfluidsynth or the
              soundfont is missing; voices are mixed but share one timbre.

Supported ABC subset:
  Header fields  L: (unit note length, e.g. 1/8)   Q: (tempo, e.g. 1/4=120 or 120)
                 M: (meter, informational)          K: (key signature -> accidentals)
  Voices         V:id [name="..."] starts a voice; all voices play in
                 parallel. Instrument per voice: a "%%MIDI program N" line
                 after the V: (abcMIDI extension), else guessed from the
                 voice name (guitar, bass, violin, ...; "drum" -> percussion),
                 else the tune-level "%%MIDI program N" from the header.
  Body           notes A-G / a-g, octave marks ' (up) and , (down),
                 accidentals ^ (sharp) _ (flat) = (natural), durations (C2, C/2,
                 C3/2), rests z, chords [CEG] / [C2E2G2], bar lines and repeat
                 marks | |: :| (ignored, repeats are not expanded).

CLI:  python3 music.py tune.abc out.wav [--engine auto|fluidsynth|sine]
                                        [--program N] [--sf2 path.sf2]
"""

import os
import re
import sys
import wave

import numpy as np

try:
    import fluidsynth                  # optional: pyfluidsynth + libfluidsynth
except ImportError:
    fluidsynth = None

SR = 44100
_MAX_SECONDS = 180.0          # bound output length per voice (untrusted input)
_NATURAL = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Major key signatures -> letters that are sharp (+1) or flat (-1).
_SHARP_KEYS = {"G": "F", "D": "FC", "A": "FCG", "E": "FCGD", "B": "FCGDA", "F#": "FCGDAE"}
_FLAT_KEYS = {"F": "B", "Bb": "BE", "Eb": "BEA", "Ab": "BEAD", "Db": "BEADG"}

# The soundfont: GeneralUser GS (freely licensed; fetched from
# https://github.com/mrbumpy409/GeneralUser-GS by `make soundfont`). Looked for
# in the repo checkout (host) and /usr/share/soundfonts (baked into the sandbox
# image); $SOUNDFONT overrides both.
_SF2_NAME = "GeneralUser-GS.sf2"
_SF2_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox", _SF2_NAME),
    os.path.join("/usr/share/soundfonts", _SF2_NAME),
]

# GM instruments guessed from a voice's name= (first keyword hit wins, so
# "Bass_Guitar" is a bass), letting a model pick instruments by naming voices
# naturally even without %%MIDI program directives.
_NAME_PROGRAMS = [
    ("bass", 33),                      # electric bass (finger)
    ("guitar", 29),                    # overdriven guitar
    ("piano", 0), ("organ", 19), ("synth", 81), ("pad", 89),
    ("violin", 40), ("fiddle", 40), ("cello", 42), ("strings", 48),
    ("flute", 73), ("whistle", 78), ("sax", 65), ("trumpet", 56), ("horn", 60),
]
_PERCUSSION_CHANNEL = 9                # GM drum channel (0-indexed)
_MELODIC_CHANNELS = [c for c in range(16) if c != _PERCUSSION_CHANNEL]

_TOKEN_RE = re.compile(
    r"\[(?P<chord>[^\]]*)\](?P<cdur>\d*/?\d*)"          # [CEG]2  chord + duration
    r"|(?P<acc>[\^_=]*)(?P<note>[A-Ga-gz])(?P<oct>[',]*)(?P<dur>\d*/?\d*)"
)
_CHORD_NOTE_RE = re.compile(r"(?P<acc>[\^_=]*)(?P<note>[A-Ga-g])(?P<oct>[',]*)(?P<dur>\d*/?\d*)")
_MIDI_PROGRAM_RE = re.compile(r"%%MIDI\s+program\s+(?:\d+\s+)?(\d+)")
_VOICE_RE = re.compile(r"^V:\s*(\S+)")
_VOICE_NAME_RE = re.compile(r'name="([^"]*)"')


def find_soundfont():
    """Return the GeneralUser GS .sf2 path ($SOUNDFONT overrides), or None."""
    env = os.environ.get("SOUNDFONT")
    if env and os.path.isfile(env):
        return env
    for path in _SF2_PATHS:
        if os.path.isfile(path):
            return path
    return None


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


def _note_midi(acc, letter, octmarks, key_acc):
    midi = 60 + _NATURAL[letter.upper()]      # uppercase C = middle C (midi 60)
    if letter.islower():
        midi += 12
    midi += 12 * octmarks.count("'") - 12 * octmarks.count(",")
    if acc:                                    # explicit accidental overrides key
        midi += acc.count("^") - acc.count("_")  # '=' (natural) contributes 0
    else:
        midi += key_acc.get(letter.upper(), 0)
    return midi


def _midi_freq(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def _voice_events(body, key_acc, unit_len, whole_note_seconds):
    events, total = [], 0.0
    for m in _TOKEN_RE.finditer(body):
        if m.group("chord") is not None:
            notes = list(_CHORD_NOTE_RE.finditer(m.group("chord")))
            midis = [_note_midi(c.group("acc"), c.group("note"), c.group("oct"), key_acc)
                     for c in notes]
            midis = list(dict.fromkeys(midis))            # [F2F2] strikes F once
            # In-chord durations ([F2F2] = 2 units) multiply the outer one.
            inner = _parse_duration(notes[0].group("dur")) if notes else 1.0
            secs = inner * _parse_duration(m.group("cdur")) * unit_len * whole_note_seconds
        else:
            note = m.group("note")
            secs = _parse_duration(m.group("dur")) * unit_len * whole_note_seconds
            midis = [] if note == "z" else [
                _note_midi(m.group("acc"), note, m.group("oct"), key_acc)]
        if secs <= 0:
            continue
        midis = [n for n in midis if 0 <= n <= 127]
        events.append({"midi": midis, "freqs": [_midi_freq(n) for n in midis],
                       "seconds": secs})
        total += secs
        if total >= _MAX_SECONDS or len(events) > 4000:
            break
    return events, total


def parse_abc(abc):
    """Parse ABC text -> (voices, meta).

    voices: [{'id', 'name', 'program', 'drums', 'events', 'seconds'}, ...] in
    first-seen order; single-voice tunes yield one voice. Each event:
    {'midi': [ints], 'freqs': [Hz], 'seconds': float}; a rest has empty lists.
    meta['program'] is the tune-level default from "%%MIDI program N";
    meta['seconds'] is wall-clock length (voices sound simultaneously).
    """
    unit_len, beat_frac, bpm, key, program = 1 / 8, 1 / 4, 120.0, "C", 0
    voices, cur, in_body = {}, None, False

    def voice(vid):
        return voices.setdefault(vid, {"id": vid, "name": "", "program": None,
                                       "lines": []})

    for line in abc.splitlines():
        line = line.strip()
        m = _MIDI_PROGRAM_RE.match(line)
        if m:                                  # abcMIDI instrument directive
            if cur is None:
                program = min(127, int(m.group(1)))
            else:
                voice(cur)["program"] = min(127, int(m.group(1)))
            continue
        if not line or line.startswith("%"):
            continue
        m = _VOICE_RE.match(line)
        if m:                                  # V: starts/switches a voice
            cur = m.group(1)
            name = _VOICE_NAME_RE.search(line)
            if name:
                voice(cur)["name"] = name.group(1)
            else:
                voice(cur)
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
            voice(cur or "1")["lines"].append(line)

    key_acc = _key_accidentals(key)
    whole_note_seconds = (60.0 / bpm) / beat_frac

    out = []
    for v in voices.values():                  # insertion order = V: order
        v["events"], v["seconds"] = _voice_events(
            " ".join(v.pop("lines")), key_acc, unit_len, whole_note_seconds)
        lowered = v["name"].lower()
        v["drums"] = "drum" in lowered or "perc" in lowered
        if v["program"] is None and not v["drums"]:
            v["program"] = next((p for kw, p in _NAME_PROGRAMS if kw in lowered),
                                program)
        out.append(v)
    if not out:
        out = [{"id": "1", "name": "", "program": program, "drums": False,
                "events": [], "seconds": 0.0}]
    return out, {"key": key, "bpm": bpm, "unit_len": unit_len,
                 "seconds": max(v["seconds"] for v in out),
                 "program": program, "voices": len(out)}


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


def _normalize(audio):
    peak = np.max(np.abs(audio)) or 1.0
    return np.int16(audio / peak * 0.9 * 32767)


def synthesize(events):
    """Sine-engine render of one voice's events -> mono int16 samples."""
    if not events:
        return np.zeros(1, dtype=np.int16)
    audio = np.concatenate([_tone(e["freqs"], e["seconds"]) for e in events])
    return _normalize(audio)


def synthesize_voices(voices):
    """Sine-engine render of all voices mixed together -> mono int16 samples."""
    tracks = [np.concatenate([_tone(e["freqs"], e["seconds"]) for e in v["events"]])
              for v in voices if v["events"]]
    if not tracks:
        return np.zeros(1, dtype=np.int16)
    mix = np.zeros(max(len(t) for t in tracks), dtype=np.float32)
    for t in tracks:
        mix[:len(t)] += t
    return _normalize(mix)


def synthesize_fluidsynth(voices, sf2_path):
    """FluidSynth render, voices in parallel -> stereo interleaved int16.

    Each voice gets its own MIDI channel and program; drum voices go to the GM
    percussion channel. All voices are scheduled on one sample-accurate
    timeline, so they sound simultaneously.
    """
    fs = fluidsynth.Synth(samplerate=float(SR))
    try:
        sfid = fs.sfload(sf2_path)
        if sfid == -1:
            raise RuntimeError(f"could not load soundfont {sf2_path}")
        actions, end, melodic = [], 0, 0       # (sample, 1=on/0=off, chan, note)
        for v in voices:
            if v["drums"]:
                ch = _PERCUSSION_CHANNEL
                fs.program_select(ch, sfid, 128, 0)     # SF2 percussion bank
            else:
                ch = _MELODIC_CHANNELS[melodic % len(_MELODIC_CHANNELS)]
                melodic += 1
                fs.program_select(ch, sfid, 0, v["program"] or 0)
            pos = 0.0
            for e in v["events"]:
                n = e["seconds"] * SR
                # Gate at 90% of the slot so consecutive notes articulate; the
                # sample's own release fills the remainder.
                for note in e["midi"]:
                    actions.append((int(round(pos)), 1, ch, note))
                    actions.append((int(round(pos + n * 0.9)), 0, ch, note))
                pos += n
            end = max(end, int(round(pos)))
        if not actions:
            return np.zeros(2, dtype=np.int16)
        actions.sort()                         # offs (0) before ons (1) per tick
        chunks, at = [], 0
        for sample, on, ch, note in actions:
            if sample > at:
                chunks.append(fs.get_samples(sample - at))
                at = sample
            if on:
                fs.noteon(ch, note, 100)
            else:
                fs.noteoff(ch, note)
        tail = end + int(0.5 * SR)             # let the last notes ring out
        if tail > at:
            chunks.append(fs.get_samples(tail - at))
        audio = np.concatenate(chunks).astype(np.float32)
    finally:
        fs.delete()
    return _normalize(audio)


def abc_to_wav(abc, out_path, engine="auto", program=None, sf2=None):
    """Render ABC to a WAV file; returns (meta, event_count).

    engine: "fluidsynth", "sine", or "auto" (fluidsynth when pyfluidsynth and a
    soundfont are present, else sine). program forces one GM instrument on all
    melodic voices, overriding "%%MIDI program" directives and voice names;
    sf2 overrides soundfont discovery.
    """
    voices, meta = parse_abc(abc)
    sf2 = sf2 or find_soundfont()
    use_fluid = fluidsynth is not None and sf2 is not None
    if engine == "sine":
        use_fluid = False
    elif engine == "fluidsynth" and not use_fluid:
        raise RuntimeError("fluidsynth engine unavailable: "
                           + ("no .sf2 soundfont found" if fluidsynth else
                              "pyfluidsynth is not installed"))
    if program is not None:
        meta["program"] = min(127, max(0, program))
        for v in voices:
            v["program"] = meta["program"]
    if use_fluid:
        samples, channels = synthesize_fluidsynth(voices, sf2), 2
        meta["engine"], meta["soundfont"] = "fluidsynth", sf2
    else:
        samples, channels = synthesize_voices(voices), 1
        meta["engine"] = "sine"
    with wave.open(out_path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return meta, sum(len(v["events"]) for v in voices)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ABC notation -> WAV synthesizer")
    ap.add_argument("abc_file")
    ap.add_argument("out", nargs="?", default="out.wav")
    ap.add_argument("--engine", choices=["auto", "fluidsynth", "sine"], default="auto")
    ap.add_argument("--program", type=int, default=None,
                    help="force one GM instrument 0-127 on all melodic voices")
    ap.add_argument("--sf2", default=None, help="path to a .sf2 soundfont")
    a = ap.parse_args()
    meta, n = abc_to_wav(open(a.abc_file).read(), a.out,
                         engine=a.engine, program=a.program, sf2=a.sf2)
    extra = ""
    if meta["engine"] == "fluidsynth":
        extra = (f", {meta['voices']} voices" if meta["voices"] > 1 else
                 f", program {meta['program']}")
        extra += f" ({os.path.basename(meta['soundfont'])})"
    print(f"synthesized {n} notes, {meta['seconds']:.1f}s, key {meta['key']}, "
          f"{meta['bpm']:.0f} bpm [{meta['engine']}{extra}] -> {a.out}")
