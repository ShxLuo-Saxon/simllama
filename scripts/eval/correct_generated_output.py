r"""
scripts/eval/correct_generated_output.py

**Output-correction gate.** Applies every post-generation correction in ONE place,
writing a corrected copy alongside the raw directory. Everything downstream --
evaluation, stimulus building, listening tests -- reads the corrected files and
applies no further correction of its own.

Replaces fix_generated_durations.py, which did only the duration repair and left
truncation duplicated across three reader scripts (a multi-track parsing bug once
survived in one reader while being fixed in another).

Corrections, in order
---------------------
1. **1-tick duration repair.** The GRU duration head sometimes decodes a duration
   of 1 tick (10 ms at 120 BPM). This value does not occur in the training data:

       ground truth (63,875 notes from 400 sections)
           min duration     = 2 ticks
           count(dur == 1)  = 0
           1st percentile   = 12 ticks
           modal values     = 25, 50, 12, 13, 75, 100

   In affected outputs a median of ~69% of notes decode to 1 tick while onsets and
   pitches stay correct -- a decode failure, not musical content. Repaired notes are
   extended to the next onset in the same channel (the voice unit); the last note of
   a voice takes --fill_cap (50 ticks = a quarter note).

2. **Truncation to the context span.** PROTO injection stops once the target bar
   count is reached, after which generation drifts into a degenerate tail. Notes with
   onset beyond the context's last note onset are dropped.

3. **Note-off clamping.** Truncation is by ONSET, so a note starting just inside the
   cutoff can still sustain far past it (observed: 40 s of audio for a 16 s context,
   the tail being one held note). Note-offs are clamped to ctx_span + one bar, so a
   final note rings naturally without droning.

4. **Instrument unification** (opt-in, --unify_program). MIDI carries a program
   (instrument) per channel, inherited from the source score. Where two outputs are
   compared BY EAR this is a confound: in an early listening-test pool, **56% of trials
   pair two different instrument sets** (one file carried piano, choir, tuba, bass and
   fretless bass across five channels), so a listener could match timbre to the
   context instead of judging musical content. Rewriting every program_change to one
   value removes the variable. Channel 9 is left alone -- there the program byte
   selects a percussion kit, not an instrument.

   **Off by default.** It changes nothing for the objective metrics, which read
   pitch/duration/onset and never see the program byte; it matters only for audio
   rendering. Turn it on for listening-test stimuli, leave it off otherwise.

Note counts are never changed by steps 1 and 4; steps 2-3 remove notes beyond the
span.

Scope note
----------
Truncation depends on the CONTEXT, so a corrected file is valid only for the context
it was generated from. Filenames encode the context stem, so this is safe here, but
the corrected directory cannot be reused against different contexts.

Usage:
    # for evaluation -- instruments untouched
    .venv\Scripts\python.exe scripts\eval\correct_generated_output.py `
        --in_dir       outputs\generated\m1_triplet_raw `
        --out_dir      outputs\generated\m1_triplet `
        --sections_dir "data\sections"

    # for listening stimuli -- everything on piano
    .venv\Scripts\python.exe scripts\eval\correct_generated_output.py `
        --in_dir       outputs\generated\m1_triplet_raw `
        --out_dir      outputs\generated\m1_triplet_audio `
        --sections_dir "data\sections" `
        --unify_program 0

Rendering to audio is a separate step, outside this repo:
    py <path to mid_to_audio.py> `
        --batch-dir <mid_dir> <out_dir> --format mp3
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mido
import numpy as np

TICKS_PER_BEAT = 50      # dataset convention: 120 BPM, 1 tick = 0.01 s
TICKS_PER_BAR = 200      # 4/4 at 120 BPM

# Ground-truth duration floor is 2 ticks; anything under 3 is a decode failure.
DEFAULT_MIN_DUR = 3
# Duration for a repaired note with no later onset in its voice. 50 ticks is the
# 2nd most common ground-truth duration (13,801 of 63,875 notes).
DEFAULT_FILL_CAP = 50
# Fallback when a note_on has no matching note_off.
UNCLOSED_NOTE_DUR = 25


def context_stem(filename: str) -> str | None:
    """Recover the context stem from a generated filename.

    e.g. '00007_s0_A_1-8_cfg1p5_high_generated.mid' -> '00007_s0_A_1-8'
         '00007_s0_A_1-8_m1_high_generated.mid'     -> '00007_s0_A_1-8'
         '00007_s0_A_1-8_context.mid'               -> '00007_s0_A_1-8'
    """
    name = filename.removesuffix(".mid")
    m = re.match(r"^(\d+_s\d+_[A-Za-z]+_[\d-]+)", name)
    return m.group(1) if m else None


def _normalize_onsets(t: np.ndarray) -> np.ndarray:
    """
    Zero the first onset, and SORT BY ONSET.

    The sort is not cosmetic. Two contexts in the section corpus are stored out
    of onset order (00517_s1_A_21-28, 00131_s1_B_21-36), and every caller reads
    the context span as ``t[-1, 0]`` -- the last ROW, not the latest onset. On
    an unsorted array those differ: 00517 reports a span 99 ticks short of its
    true extent, 00131 by 965 ticks (~10 s). Everything downstream then trusts
    a boundary that is inside the music, so truncation never removes the tail
    and the clip runs past its stated length.

    Sorting here makes ``t[0, 0]`` and ``t[-1, 0]`` mean what every caller
    already assumes they mean.
    """
    if len(t) == 0:
        return t
    t = t[np.argsort(t[:, 0], kind="stable")].copy()
    t[:, 0] -= t[0, 0]
    return t


def midi_to_events(path: Path) -> tuple[list[dict], int, list[mido.Message]]:
    """Read a MIDI file into note dicts with absolute ticks.

    Reads EVERY track: some generated files place a lone note in track 0 and the
    real content in track 1, so reading track 0 alone silently loses ~100 notes.
    """
    mid = mido.MidiFile(str(path))
    notes: list[dict] = []
    programs: list[mido.Message] = []
    for track in mid.tracks:
        t = 0
        open_notes: dict[tuple[int, int], dict] = {}
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                d = {"onset": t, "pitch": msg.note, "vel": msg.velocity,
                     "ch": msg.channel, "dur": None}
                notes.append(d)
                open_notes[(msg.note, msg.channel)] = d
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                k = (msg.note, msg.channel)
                if k in open_notes:
                    d = open_notes.pop(k)
                    d["dur"] = max(1, t - d["onset"])
            elif msg.type == "program_change":
                programs.append(msg.copy(time=0))
    for d in notes:
        if d["dur"] is None:
            d["dur"] = UNCLOSED_NOTE_DUR
    notes.sort(key=lambda d: d["onset"])
    return notes, mid.ticks_per_beat, programs


def fix_durations(notes: list[dict], min_dur: int, fill_cap: int) -> int:
    """Extend sub-minimum notes to the next onset in the same CHANNEL.

    Channel is the voice unit, so a repaired note runs until the voice next sounds.
    The gap is filled UNCAPPED -- it is real musical time, and clipping it would
    leave an unintended rest. Chord partners share an onset, so the search skips to
    the next STRICTLY later onset; otherwise a chord note would be clipped to zero.
    `fill_cap` applies only to the final note of a voice.
    """
    by_channel: dict[int, list[dict]] = {}
    for n in notes:
        by_channel.setdefault(n["ch"], []).append(n)
    for group in by_channel.values():
        group.sort(key=lambda d: d["onset"])

    n_fixed = 0
    for group in by_channel.values():
        onsets = [d["onset"] for d in group]
        for i, n in enumerate(group):
            if n["dur"] >= min_dur:
                continue
            nxt = next((onsets[j] for j in range(i + 1, len(group))
                        if onsets[j] > n["onset"]), None)
            n["dur"] = max(min_dur, nxt - n["onset"]) if nxt is not None else fill_cap
            n_fixed += 1
    return n_fixed


def truncate_and_clamp(notes: list[dict], ctx_span: int,
                       tail: int = TICKS_PER_BAR) -> list[dict]:
    """Drop notes whose onset is past ctx_span; clamp note-offs to ctx_span + tail.

    Onsets are re-zeroed first, matching the convention used when the metric is
    computed (first note at tick 0).
    """
    if not notes:
        return notes
    t0 = notes[0]["onset"]
    out = []
    for n in notes:
        onset = n["onset"] - t0
        if onset > ctx_span:
            continue
        off = min(onset + n["dur"], ctx_span + tail)
        if off <= onset:
            continue
        out.append({**n, "onset": onset, "dur": off - onset})
    return out


def unify_programs(programs: list[mido.Message],
                   program: int) -> list[mido.Message]:
    """
    Rewrite every program_change to a single GM program, one per channel.

    Channel 9 is left alone: on that channel the program byte selects a
    percussion kit rather than an instrument.
    """
    out, seen = [], set()
    for m in programs:
        if m.channel in seen:
            continue
        seen.add(m.channel)
        out.append(m.copy(time=0) if m.channel == 9
                   else m.copy(time=0, program=program))
    return out


def events_to_midi(notes: list[dict], tpb: int,
                   programs: list[mido.Message] | None = None) -> mido.MidiFile:
    """Write notes to a single track, preserving program_change messages."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    seen = set()
    for m in (programs or []):
        key = (m.channel, m.program)
        if key not in seen:
            seen.add(key)
            track.append(m.copy(time=0))
    events = []
    for n in notes:
        events.append((n["onset"], 1, "note_on", n["pitch"], n["vel"], n["ch"]))
        events.append((n["onset"] + n["dur"], 0, "note_off", n["pitch"], 0, n["ch"]))
    # note_off before note_on at the same tick, so a repeated pitch retriggers
    events.sort(key=lambda e: (e[0], e[1]))
    prev = 0
    for tick, _, typ, pitch, vel, ch in events:
        track.append(mido.Message(typ, note=pitch, velocity=vel,
                                  channel=ch, time=max(0, tick - prev)))
        prev = tick
    return mid


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--sections_dir", required=True,
                   help="Context .npy files -- needed to compute each context span")
    p.add_argument("--min_dur", type=int, default=DEFAULT_MIN_DUR,
                   help="Durations below this are treated as decode failures")
    p.add_argument("--fill_cap", type=int, default=DEFAULT_FILL_CAP,
                   help="Duration for a repaired note with no later onset in its "
                        "voice (50 = quarter note at 120 BPM)")
    p.add_argument("--tail", type=int, default=TICKS_PER_BAR,
                   help="Ticks past ctx_span a note may ring (default 1 bar)")
    p.add_argument("--no_truncate", action="store_true",
                   help="Repair durations only; skip truncation and clamping. "
                        "For files that are not context-relative (e.g. contexts).")
    p.add_argument("--unify_program", type=int, metavar="GM", default=None,
                   help="Rewrite every program_change to this GM program "
                        "(0 = acoustic grand piano). Removes instrument as a "
                        "variable when outputs are compared by ear -- 56%% of "
                        "listening-test trials otherwise pair two DIFFERENT "
                        "instrument sets, letting a listener match timbre "
                        "instead of musical content. Percussion (ch 9) is "
                        "left alone.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    sec_dir = Path(args.sections_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Not a directory: {in_dir}")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Context span per stem, cached
    spans: dict[str, int] = {}

    def ctx_span_for(stem: str) -> int | None:
        if stem not in spans:
            f = sec_dir / f"{stem}.npy"
            if not f.exists():
                spans[stem] = -1
            else:
                arr = _normalize_onsets(np.load(str(f)).astype(np.int64))
                spans[stem] = int(arr[-1, 0]) if len(arr) else -1
        return spans[stem] if spans[stem] >= 0 else None

    files = sorted(in_dir.glob("*.mid"))
    print(f"Source : {in_dir}  ({len(files)} files)")
    print(f"Target : {out_dir}{'  [DRY RUN]' if args.dry_run else ''}")
    steps = ["1-tick repair"]
    if not args.no_truncate:
        steps += ["truncate to ctx_span", f"clamp +{args.tail} ticks"]
    if args.unify_program is not None:
        steps += [f"unify instrument -> GM {args.unify_program}"]
    print(f"Steps  : {'  ->  '.join(steps)}\n")

    n_repaired = n_notes = n_dropped = n_touched = n_missing = n_unified = 0
    for f in files:
        try:
            notes, tpb, programs = midi_to_events(f)
        except Exception as e:                       # noqa: BLE001
            print(f"  SKIP {f.name}: {e}")
            continue
        if not notes:
            if not args.dry_run:
                shutil.copy2(f, out_dir / f.name)
            continue

        before = len(notes)
        fixed = fix_durations(notes, args.min_dur, args.fill_cap)

        if not args.no_truncate:
            stem = context_stem(f.name)
            span = ctx_span_for(stem) if stem else None
            if span is None:
                n_missing += 1
            else:
                notes = truncate_and_clamp(notes, span, args.tail)

        n_notes += before
        n_repaired += fixed
        n_dropped += before - len(notes)
        if fixed or before != len(notes):
            n_touched += 1
        if args.unify_program is not None:
            n_progs = len({m.program for m in programs if m.channel != 9})
            if n_progs > 1:
                n_unified += 1
            programs = unify_programs(programs, args.unify_program)

        if not args.dry_run:
            events_to_midi(notes, tpb, programs).save(str(out_dir / f.name))

    if args.unify_program is not None:
        print(f"Instrument       : all -> GM {args.unify_program}  "
              f"({n_unified} files had >1 instrument)")
    print(f"Files written    : {len(files)}  (modified: {n_touched})")
    print(f"Notes in         : {n_notes}")
    print(f"  durations fixed: {n_repaired} ({n_repaired / max(n_notes, 1):.1%})")
    if not args.no_truncate:
        print(f"  dropped by trunc: {n_dropped} ({n_dropped / max(n_notes, 1):.1%})")
    if n_missing:
        print(f"  WARNING: {n_missing} files had no matching context .npy -- "
              f"left untruncated")


if __name__ == "__main__":
    main()
