r"""
scripts/eval/make_short_stimuli.py

Build a shortened, **audio-only** copy of a stimulus set.

Why
---
The full clips run to a 16 s median (23.8 s at the tail), which makes a 17-trial
session long: three clips per trial plus response time. A shorter cut tests the
same trials in roughly half the listening time.

The window-sensitivity check (`eval_window_sensitivity.py`)
found the objective effect survives truncation --
cutting context and both outputs to 8 s still gives fp 0.563, p 0.012 -- so a
shortened set is scoring the same phenomenon, not a different one.

What it does
------------
Trims each clip to the first N seconds by cutting the MIDI and re-rendering,
rather than cutting the MP3. Re-rendering avoids a mid-note audio chop: MIDI
truncation drops note-ONs past the cut and lets the remaining notes finish, so
the clip ends on sounding notes instead of a click.

Clips already shorter than the window are copied through unchanged.

Trailing silence
----------------
MuseScore imports MIDI as a *score*: it pads the last bar with rests to a full
measure and adds a reverb tail. Every render therefore ends with ~2-3 s of
silence (measured: sound stops at 11.0 s, file runs to 13.1 s). That silence is
dead listening time on every clip -- three clips a trial, 17 trials -- so each
MP3 is trimmed with ffmpeg after rendering, keeping a short pad so the last
note's decay is not cut.

**Audio only.** The output holds MP3s and the answer key; no MIDI. The MIDI
source of record stays in the full-length set, so there is one place to go back
to and no second copy to drift.

Usage:
    .venv\Scripts\python.exe scripts\eval\make_short_stimuli.py --seconds 10
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mido

# the renderer (a copy ships beside this script in the release); MID_TO_AUDIO overrides
RENDER = Path(os.environ.get("MID_TO_AUDIO", Path(__file__).resolve().parent / "mid_to_audio.py"))
TICKS_PER_BEAT = 50          # dataset convention: 120 BPM, 1 tick = 0.01 s

# ffmpeg is installed via WinGet but not on PATH; fall back to PATH if moved.
_FF_WINGET = Path.home() / ("AppData/Local/Microsoft/WinGet/Packages/"
                            "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
                            "ffmpeg-8.1.1-full_build/bin/ffmpeg.exe")
FFMPEG = str(_FF_WINGET) if _FF_WINGET.exists() else (shutil.which("ffmpeg") or "")

SILENCE_DB = -50             # below this counts as silence
KEEP_PAD = 0.3               # seconds of decay kept after the last sound


def truncate_midi(src: Path, dst: Path, seconds: float,
                  tail: float = 1.0) -> bool:
    """Write src cut to `seconds`. Returns True if anything was removed.

    Two things have to happen, and doing only the first leaves the file the
    original length:

      1. drop note-ONs at or after the cut (and their matching note-OFFs, or
         the file ends with hanging notes);
      2. **clamp the note-OFFs of notes still sounding at the cut** to
         `seconds + tail`. Without this a long note started at 9 s rings to its
         original end -- the MIDI still reports 16 s and renders that long.

    `tail` lets the final notes decay rather than stopping dead on the cut.
    """
    mid = mido.MidiFile(str(src))
    if mid.length <= seconds:
        shutil.copy2(src, dst)
        return False

    scale = mid.ticks_per_beat / TICKS_PER_BEAT
    cut = int(seconds * 100 * scale)
    end = int((seconds + tail) * 100 * scale)

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for track in mid.tracks:
        events = []                              # (abs_tick, msg)
        t = 0
        dropped: set[tuple[int, int]] = set()
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                if t >= cut:
                    dropped.add((msg.note, msg.channel))
                    continue
                events.append((t, msg))
            elif msg.type == "note_off" or (msg.type == "note_on"
                                            and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in dropped:
                    dropped.discard(key)
                    continue
                events.append((min(t, end), msg))   # clamp a ringing note
            else:
                if t <= end:
                    events.append((t, msg))

        events.sort(key=lambda e: e[0])
        new_track = mido.MidiTrack()
        prev = 0
        for abs_t, msg in events:
            new_track.append(msg.copy(time=abs_t - prev))
            prev = abs_t
        out.tracks.append(new_track)

    out.save(str(dst))
    return True


def trim_trailing_silence(mp3: Path) -> None:
    """Strip trailing silence in place, keeping KEEP_PAD seconds of decay.

    Reverse -> remove leading silence -> reverse. The pad is added back with
    apad so the final note's decay is not clipped to a hard stop.
    """
    tmp = mp3.with_suffix(".trim.mp3")
    af = (f"areverse,silenceremove=start_periods=1:"
          f"start_threshold={SILENCE_DB}dB:start_silence={KEEP_PAD},areverse")
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(mp3), "-af", af, "-b:a", "160k", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg trim failed on {mp3.name}: {r.stderr[:200]}")
    tmp.replace(mp3)


def trim_leading_silence(mp3: Path, keep: float = 0.05) -> None:
    """Strip silence before the first sound, in place.

    Every clip starts on a note (onsets are normalised to 0), so silence at the
    start is a render artefact: MuseScore imports MIDI as a score, and for some
    files it guesses a pickup bar and pads before it (seen: 1.1 s before
    00551_s0_A_4-10 sim:high, reproducible on re-render). A late start shifts
    the clip against the reference, so it is removed.
    """
    tmp = mp3.with_suffix(".lead.mp3")
    af = (f"silenceremove=start_periods=1:start_threshold={SILENCE_DB}dB:"
          f"start_silence={keep}")
    r = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(mp3), "-af", af, "-b:a", "160k", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg lead trim failed on {mp3.name}: {r.stderr[:200]}")
    tmp.replace(mp3)


def leading_silence(mp3: Path) -> float:
    """Seconds of silence before the first sound (0 if it starts on sound)."""
    import re
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(mp3), "-af",
                          f"silencedetect=noise={SILENCE_DB}dB:d=0.2", "-f",
                          "null", "-"], capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    if starts and starts[0] < 0.01 and ends:
        return ends[0]
    return 0.0


def peak_db(mp3: Path) -> float:
    """Peak level in dB. A render can succeed and still be silent."""
    import re
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(mp3), "-af",
                          "volumedetect", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"max_volume: (-?[\d.]+)", err)
    return float(m.group(1)) if m else -999.0


SILENT_DB = -40              # a clip whose peak is below this is a failed render


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--src", default="listening_test/stimuli/test1_two_sets_full")
    ap.add_argument("--out", default=None,
                    help="Output dir (default: stimuli/build/short_<N>s_two_sets)")
    ap.add_argument("--name", default=None,
                    help="Subdirectory name (default: short_<N>s)")
    ap.add_argument("--format", default="mp3")
    args = ap.parse_args()

    src_root = Path(args.src)
    n = int(args.seconds) if args.seconds == int(args.seconds) else args.seconds
    out_root = (Path(args.out) if args.out else
                Path("listening_test/stimuli/build") / (args.name or f"short_{n}s_two_sets"))

    sets = sorted(d for d in src_root.glob("set_*") if d.is_dir())
    if not sets:
        sys.exit(f"No set_* directories in {src_root}")

    print(f"Source : {src_root}")
    print(f"Target : {out_root}   (cut to {args.seconds:g}s, "
          f"{args.format} only)\n")

    for sdir in sets:
        trials = sdir / "trials"
        mids = sorted(trials.glob("*.mid"))
        if not mids:
            continue
        dst = out_root / sdir.name / "trials"
        dst.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            n_cut = 0
            for f in mids:
                if truncate_midi(f, tmp / f.name, args.seconds):
                    n_cut += 1
            print(f"{sdir.name}: {len(mids)} clips, {n_cut} truncated, "
                  f"{len(mids) - n_cut} already shorter")
            r = subprocess.run(
                [sys.executable, str(RENDER), "--batch-dir", str(tmp),
                 str(dst), "--format", args.format, "--overwrite"],
                capture_output=True, text=True)
            print("   " + (r.stdout.strip().splitlines() or ["(no output)"])[-1])
            if r.returncode != 0:
                print("   RENDER FAILED:", r.stderr.strip()[:300])

        if not FFMPEG:
            print("   WARNING: ffmpeg not found -- trailing silence NOT trimmed")
        else:
            for mp3 in sorted(dst.glob(f"*.{args.format}")):
                if leading_silence(mp3) > 0.2:
                    trim_leading_silence(mp3)
                trim_trailing_silence(mp3)
            print(f"   trimmed trailing silence ({SILENCE_DB} dB, "
                  f"{KEEP_PAD}s pad kept)")
            # "Render ok" only means a non-empty file. Two clips once rendered
            # as pure silence (-91 dB, velocity-0 MIDI) and passed that check,
            # then went live in the study. Refuse to finish on any silent clip.
            silent = [m.name for m in sorted(dst.glob(f"*.{args.format}"))
                      if peak_db(m) < SILENT_DB]
            if silent:
                sys.exit(f"   SILENT CLIPS (peak < {SILENT_DB} dB): {silent}")
            print(f"   all clips audible (peak >= {SILENT_DB} dB)")

        # the answer key travels with the audio; nothing else does
        key = sdir / "answer_key.csv"
        if key.exists():
            shutil.copy2(key, out_root / sdir.name / "answer_key.csv")

    print(f"\nWrote {out_root}")
    for d in sorted(out_root.glob("set_*")):
        print(f"  {d.name}: {len(list((d/'trials').glob('*.'+args.format)))} "
              f"{args.format}")


if __name__ == "__main__":
    main()
