# src/generation/midi_export.py
"""
MIDI Export Utilities
======================
Converts piano-roll arrays and token sequences to .mid files.
"""

import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (PITCH_MIN, N_PITCHES, STEPS_PER_BEAT,
                         DEFAULT_TEMPO)

try:
    import pretty_midi
    HAS_PM = True
except ImportError:
    HAS_PM = False


def pianoroll_to_midi(roll: np.ndarray,
                       out_path: str,
                       tempo: float = DEFAULT_TEMPO,
                       threshold: float = 0.1) -> None:
    """
    Convert a (T, 88) float32 piano-roll to a MIDI file.

    Args:
        roll      : (T, 88) values in [0,1], velocity normalised
        out_path  : output .mid file path
        tempo     : BPM
        threshold : minimum activation to count as a note
    """
    if not HAS_PM:
        raise ImportError("pretty_midi required: pip install pretty_midi")

    pm        = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano     = pretty_midi.Instrument(program=0, name="Piano")
    step_dur  = 60.0 / tempo / STEPS_PER_BEAT
    T, P      = roll.shape
    active    = {}   # pitch → (start_time, velocity)

    for t in range(T):
        for p in range(min(P, N_PITCHES)):
            pitch = p + PITCH_MIN
            val   = float(roll[t, p])
            time  = t * step_dur
            if val >= threshold:
                if pitch not in active:
                    active[pitch] = (time, max(1, int(val * 127)))
            else:
                if pitch in active:
                    s, v = active.pop(pitch)
                    if time > s:
                        piano.notes.append(
                            pretty_midi.Note(v, pitch, s, time))

    end_time = T * step_dur
    for pitch, (s, v) in active.items():
        piano.notes.append(pretty_midi.Note(v, pitch, s, end_time))

    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pm.write(out_path)


def events_to_midi(events: list,
                    out_path: str,
                    tempo: float = DEFAULT_TEMPO) -> None:
    """
    Convert a list of note-event dicts to a MIDI file.
    Each event: {pitch, velocity, start_sec, duration_sec}
    """
    if not HAS_PM:
        raise ImportError("pretty_midi required")

    pm    = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    t_cur = 0.0

    for e in events:
        start = t_cur + e.get("start_sec", 0.0)
        end   = start + max(0.05, e.get("duration_sec", 0.25))
        piano.notes.append(
            pretty_midi.Note(
                max(1, min(127, e.get("velocity", 80))),
                max(0, min(127, e.get("pitch", 60))),
                start, end
            )
        )
        t_cur = start + max(0.05, e.get("duration_sec", 0.25)) * 0.9

    pm.instruments.append(piano)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pm.write(out_path)