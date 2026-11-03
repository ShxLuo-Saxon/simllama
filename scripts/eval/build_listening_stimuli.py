r"""
scripts/eval/build_listening_stimuli.py

**Selection and rendering only.** Reads the triage table (which has already
measured, filtered and stratified every context) and writes a stimulus set.

  triage_listening_stimuli.py  ->  measurement + filtering + strata
  THIS SCRIPT                  ->  select per stratum, build trials, render MIDI

Audio is read from the CORRECTED generation directory
(scripts/eval/correct_generated_output.py), which already applies the duration
repair, truncation to the context span, and note-off clamping. **No correction or
truncation happens here.**

Design (listening_test/protocol.md)
----------------------------------------
  15 test trials  +  2 pinned attention checks  =  17 trials, ~20 min

  P-pos : positive delta. Q1 is tested on these only.
  N-neg : negative delta. Metric-divergence probe, reported separately.

  Default allocation 12 / 3 per set. Q1 power depends on the POSITIVE count:
  12 positive trials x 20 valid participants = 240 observations = 0.80 power at a
  true rate of 0.58 (threshold: 134 of 240 correct). Recruit 25 for 20 valid.

  Sign-only strata: no median split (Q2b supersedes the delta-slope test) and no
  dead band (excluding near-zero delta would filter out the metric's own failures).

  Two questions per trial:
    Q1  which output is more similar to the context?      -> binary, vs chance
    Q2b how similar is the CHOSEN output to the context?  -> 1-3, vs cs(ctx, chosen)

  --n_sets 2 builds disjoint rotating sets so more contexts are covered without
  lengthening the session. Attention checks are identical across sets.

Scoring is the same in every stratum: did the listener choose the `sim:high`
output? Never score against the metric.

Outputs (per set):
    trials/TrialNN_1_Context.mid   (sorts in playback order)
    trials/TrialNN_2_A.mid
    trials/TrialNN_3_B.mid
    answer_key.csv       <- correct answer AND provenance (stratum, delta, seed,
                            source stems). Do not open until responses are in.
    response_sheet.csv   <- blank

The triage table lives in the same build directory, so each build records the exact
frame it selected from.

Filenames carry NO information about condition.

Usage:
    .venv\Scripts\python.exe scripts\eval\build_listening_stimuli.py `
        --triage       outputs\listening_test\formal\triage_cfg1p5.csv `
        --gen_dir      outputs\generated\m1_listening_candidates `
        --sections_dir "data\sections" `
        --pairs_csv    "data\sections\pairs.csv" `
        --out_dir      outputs\listening_test\formal\study `
        --n_sets 2 --alloc "P-pos=12,N-neg=3" --seed 11
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

import mido  # noqa: E402

from scripts.eval.eval_triplet import compound_sim  # noqa: E402

TICKS_PER_BEAT = 50
MIN_NOTES = 10

# ── Pinned attention checks ───────────────────────────────────────────
# FIXED across participants, sets and rebuilds, so screening is a constant control.
#
# Both options are GROUND TRUTH (never generated): the real same-letter GT target
# for the context vs a real section from a different piece. Using a generated output
# as the "correct" option would be unsound -- its quality is what the study tests, so
# a poor generation makes the intended answer unclear (observed once:
# margin +0.085, indistinguishable to an expert listener).
#
# Contexts come from the pool EXCLUDED by the quality filters, so no usable test
# stimulus is spent. Selected for a large similarity margin and a foil that spans the
# context, so the answer cannot be guessed from length.
PINNED_CHECKS = [
    # (context, GT target: same piece + same section letter, foil: different piece)
    ("00498_s3_A_39-46", "00498_s0_A_15-22", "00309_s1_A_6-13"),   # margin +0.364
    ("00054_s1_B_11-18", "00054_s4_B_39-46", "00433_s0_A_6-13"),   # margin +0.311
]

STRATA = ["P-pos", "N-neg"]


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


def npy_to_midi(tokens: np.ndarray, snap: bool = False) -> mido.MidiFile:
    """Compound-token array -> MidiFile. No clamping: already done upstream.

    snap=True repairs the sixteenth-note grid. At ticks_per_beat=50 a sixteenth
    is 12.5 ticks, which the integer corpus stores as 12 or 13. In sparse music
    that is inaudible, but where several voices run in sixteenths the rounding
    puts them out of step: onsets that should coincide land 5-7 ticks (50-70 ms)
    apart and the passage smears. Verified against the source score for
    00517_s1_A_21-28 (MusicXML 1064.mxl, bars 21-28): every note offset is a
    multiple of 1/4 quarterLength -- a clean sixteenth grid with no triplets --
    so the irregularity is a storage artefact, not the music.

    Snapping rounds each onset to the nearest true 12.5-tick position and writes
    at DOUBLE resolution so the result is exact in integer ticks. Mean shift on
    that section is 0.86 ticks (8.6 ms), max 6.

    Off by default: it is a rendering repair for listening stimuli and must not
    silently alter anything the objective metrics are computed from.
    """
    tokens = _normalize_onsets(tokens.astype(np.int64))
    tpb = TICKS_PER_BEAT
    if snap:
        # 12.5-tick grid at tpb=50 -> 25 ticks at tpb=100, exact in integers
        tpb = TICKS_PER_BEAT * 2
        scaled = tokens.astype(np.float64) * 1.0
        scaled[:, 0] = np.round(tokens[:, 0] / 12.5) * 25.0
        scaled[:, 1] = np.maximum(1.0, np.round(tokens[:, 1] / 12.5) * 25.0)
        tokens = scaled.astype(np.int64)
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    events = []
    for row in tokens:
        onset, dur, octave, pc, ch, vel = row[:6]
        pitch = int(octave) * 12 + int(pc)
        if not (0 <= pitch <= 127):
            continue
        events.append((int(onset), 1, "note_on", pitch,
                       max(1, min(127, int(vel))), min(int(ch), 15)))
        events.append((int(onset) + max(1, int(dur)), 0, "note_off", pitch,
                       0, min(int(ch), 15)))
    events.sort(key=lambda e: (e[0], e[1]))
    prev = 0
    for tick, _, typ, pitch, vel, ch in events:
        track.append(mido.Message(typ, note=pitch, velocity=vel,
                                  channel=ch, time=max(0, tick - prev)))
        prev = tick
    return mid


def read_gen(path: Path) -> np.ndarray | None:
    """Read a corrected generated MIDI to compound tokens. No truncation."""
    if not path.exists():
        return None
    mid = mido.MidiFile(str(path))
    rows: list[list[int]] = []
    for track in mid.tracks:               # read EVERY track
        t = 0
        on: dict[tuple[int, int], tuple[int, int]] = {}
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on[(msg.note, msg.channel)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                k = (msg.note, msg.channel)
                if k in on:
                    st, vel = on.pop(k)
                    rows.append([st, max(1, t - st), msg.note // 12,
                                 msg.note % 12, msg.channel, vel])
    if len(rows) < MIN_NOTES:
        return None
    return _normalize_onsets(np.array(sorted(rows, key=lambda r: r[0]), dtype=np.int64))


def max_run(seq: list) -> int:
    mx = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        mx = max(mx, cur)
    return mx if seq else 0


def build_set(sel: pd.DataFrame, checks: list[dict], gen_dir: Path, sec_dir: Path,
              out_dir: Path, tag: str, rng: random.Random, seed: int) -> pd.DataFrame:
    """Render one stimulus set. Returns the answer key."""
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # cs_high / cs_low are carried into the key so the scorer can compute
    # cs(ctx, chosen) for Q2b without re-reading the triage table.
    trials = [{"kind": "test", "ctx_stem": r.ctx_stem, "delta": r.delta,
               "stratum": r.stratum, "cs_high": r.cs_high, "cs_low": r.cs_low,
               "gt_stem": None, "foil_stem": None}
              for r in sel.itertuples()]
    trials += checks

    # Balance A/B WITHIN each stratum, not just globally: a global-only balance can
    # still correlate slot with stratum, letting a participant slot-bias masquerade
    # as a per-stratum effect.
    by_stratum: dict[str, list[dict]] = {}
    for tr in trials:
        by_stratum.setdefault(tr["stratum"], []).append(tr)
    for group in by_stratum.values():
        k = len(group)
        slots = ["A"] * (k // 2) + ["B"] * (k - k // 2)
        rng.shuffle(slots)
        for tr, s in zip(group, slots):
            tr["_slot"] = s

    # Order so neither the slot nor the stratum sequence has a long run -- listeners
    # notice streaks and may adopt a strategy.
    best = None
    for _ in range(2000):
        rng.shuffle(trials)
        score = max(max_run([t["_slot"] for t in trials]),
                    max_run([t["stratum"] for t in trials]))
        if best is None or score < best[0]:
            best = (score, list(trials))
        if score <= 2:
            break
    trials = best[1]

    key_rows, sheet_rows, skipped = [], [], []
    n = 0
    for tr in trials:
        stem = tr["ctx_stem"]
        ctx_npy = sec_dir / f"{stem}.npy"
        if not ctx_npy.exists():
            skipped.append((stem, "context npy missing"))
            continue
        ctx = _normalize_onsets(np.load(str(ctx_npy)).astype(np.int64))
        cut = int(ctx[-1, 0])

        if tr["kind"] == "test":
            hi = read_gen(gen_dir / f"{stem}_{tag}_high_generated.mid")
            lo = read_gen(gen_dir / f"{stem}_{tag}_low_generated.mid")
            if hi is None or lo is None:
                skipped.append((stem, "output unreadable"))
                continue
            opt_high, opt_other, other_desc = hi, lo, "sim:low"
        else:
            gt = _normalize_onsets(np.load(str(sec_dir / f"{tr['gt_stem']}.npy")).astype(np.int64))
            fo = _normalize_onsets(np.load(str(sec_dir / f"{tr['foil_stem']}.npy")).astype(np.int64))
            opt_high, opt_other = gt[gt[:, 0] <= cut], fo[fo[:, 0] <= cut]
            other_desc = "GT from unrelated piece"
            if len(opt_high) < MIN_NOTES or len(opt_other) < MIN_NOTES:
                skipped.append((stem, "check option too short"))
                continue

        n += 1
        t = f"Trial{n:02d}"
        high_slot = tr["_slot"]
        a_arr = opt_high if high_slot == "A" else opt_other
        b_arr = opt_other if high_slot == "A" else opt_high

        # Numeric prefix keeps playback order when files are sorted alphabetically
        npy_to_midi(ctx).save(str(trials_dir / f"{t}_1_Context.mid"))
        npy_to_midi(a_arr).save(str(trials_dir / f"{t}_2_A.mid"))
        npy_to_midi(b_arr).save(str(trials_dir / f"{t}_3_B.mid"))

        key_rows.append({
            "trial": t, "kind": tr["kind"], "stratum": tr["stratum"],
            "ctx_stem": stem, "delta": tr["delta"],
            "cs_high": tr["cs_high"], "cs_low": tr["cs_low"],
            "high_slot": high_slot, "other_is": other_desc,
            "gt_stem": tr["gt_stem"], "foil_stem": tr["foil_stem"],
            "ctx_sec": round(cut / 100.0, 2), "seed": seed,
        })
        sheet_rows.append({"participant_id": "", "trial": t,
                           "answer_A_or_B": "", "similarity_1to3": "", "notes": ""})

    key = pd.DataFrame(key_rows)
    # answer_key.csv IS the manifest -- it carries stratum, delta, seed and source
    # stems alongside the correct answer. No separate manifest file.
    key.to_csv(out_dir / "answer_key.csv", index=False)
    pd.DataFrame(sheet_rows).to_csv(out_dir / "response_sheet.csv", index=False)
    if skipped:
        print(f"    skipped: {skipped}")
    return key


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--triage", default="listening_test/stimuli/build/triage_m1.csv")
    p.add_argument("--gen_dir", required=True,
                   help="CORRECTED generation dir")
    p.add_argument("--sections_dir", required=True)
    p.add_argument("--pairs_csv", default=None,
                   help="pairs.csv -- only needed to verify pinned checks")
    # Defaults to build/rebuild so a repeated run can never overwrite the live test
    # stimuli in stimuli/test1_* or stimuli/test2_*.
    p.add_argument("--out_dir", default="listening_test/stimuli/build/rebuild")
    p.add_argument("--alloc", default="P-pos=12,N-neg=3",
                   help="Contexts per stratum per set")
    p.add_argument("--n_sets", type=int, default=2,
                   help="Disjoint rotating sets. More contexts covered, same session "
                        "length. Checks are identical across sets.")
    p.add_argument("--seed", type=int, default=11,
                   help="Recorded in answer_key.csv. The draw is made ONCE and "
                        "used as drawn -- see --random_draw.")
    p.add_argument("--random_draw", action="store_true",
                   help="Uniform draw from the whole pool, ignoring --alloc "
                        "(protocol section 3.3). The positive/negative mix is "
                        "whatever the draw gives.")
    p.add_argument("--n_trials", type=int, default=15,
                   help="Test trials per set under --random_draw")
    p.add_argument("--tag", default="m1")
    args = p.parse_args()

    gen_dir, sec_dir = Path(args.gen_dir), Path(args.sections_dir)
    out_root = Path(args.out_dir)
    rng = random.Random(args.seed)

    df = pd.read_csv(args.triage)
    print(f"Triage: {len(df)} contexts")

    if args.random_draw:
        # ── Uniform draw (protocol section 3.3) ───────────────────────────────
        # One shuffle of the WHOLE pool, then consecutive blocks per set. The
        # positive/negative mix is whatever the draw gives; it is reported, not
        # controlled, and never re-drawn -- re-drawing on a composition the
        # experimenter dislikes is stratified allocation described as random.
        need = args.n_trials * args.n_sets
        if len(df) < need:
            raise SystemExit(f"Pool has {len(df)}, need {need} for "
                             f"{args.n_sets} sets of {args.n_trials}.")
        pool = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        blocks = [pool.iloc[i * args.n_trials:(i + 1) * args.n_trials]
                  for i in range(args.n_sets)]
        draws = None
        print(f"  uniform draw of {args.n_trials} x {args.n_sets} "
              f"from {len(df)} (seed {args.seed})")
        exp_pos = args.n_trials * (df.stratum == "P-pos").mean()
        for i, b in enumerate(blocks):
            npos = int((b.stratum == "P-pos").sum())
            print(f"    set_{chr(ord('A')+i)}: {npos} positive / "
                  f"{len(b)-npos} negative   (expected {exp_pos:.1f} positive)")
    else:
        # ── Stratified allocation (superseded; retained for comparison) ───────
        alloc = {k.strip(): int(v) for k, v in
                 (part.split("=") for part in args.alloc.split(","))}
        for s in STRATA:
            need = alloc.get(s, 0) * args.n_sets
            have = int((df.stratum == s).sum())
            flag = "OK" if have >= need else "SHORT"
            print(f"  {s:8s} have {have:3d}  need {need:3d}  {flag}")
            if have < need:
                raise SystemExit(f"Not enough {s} contexts for {args.n_sets} sets.")

        draws = {}
        for s in STRATA:
            pool = df[df.stratum == s].sample(frac=1, random_state=args.seed)
            draws[s] = [pool.iloc[i * alloc[s]:(i + 1) * alloc[s]]
                        for i in range(args.n_sets)]
        blocks = None

    # Pinned checks, verified once
    checks = []
    for ctx_s, gt_s, foil_s in PINNED_CHECKS:
        ca = _normalize_onsets(np.load(str(sec_dir / f"{ctx_s}.npy")).astype(np.int64))
        cut = int(ca[-1, 0])
        ga = _normalize_onsets(np.load(str(sec_dir / f"{gt_s}.npy")).astype(np.int64))
        fa = _normalize_onsets(np.load(str(sec_dir / f"{foil_s}.npy")).astype(np.int64))
        margin = compound_sim(ca, ga[ga[:, 0] <= cut]) - compound_sim(ca, fa[fa[:, 0] <= cut])
        print(f"  check {ctx_s}: margin {margin:+.3f}")
        checks.append({"kind": "check", "ctx_stem": ctx_s, "delta": np.nan,
                       "stratum": "CHECK", "cs_high": np.nan, "cs_low": np.nan,
                       "gt_stem": gt_s, "foil_stem": foil_s})

    out_root.mkdir(parents=True, exist_ok=True)
    # The triage table already lives in the build directory (triage writes it
    # there), so each build records the exact frame it selected from.
    if Path(args.triage).resolve().parent != out_root.resolve():
        pd.read_csv(args.triage).to_csv(out_root / Path(args.triage).name, index=False)

    for i in range(args.n_sets):
        name = f"set_{chr(ord('A') + i)}"
        sel = (blocks[i].reset_index(drop=True) if draws is None
               else pd.concat([draws[s][i] for s in STRATA]).reset_index(drop=True))
        out_dir = out_root / name
        print(f"\n=== {name} ===")
        key = build_set(sel, [dict(c) for c in checks], gen_dir, sec_dir,
                        out_dir, args.tag, rng, args.seed)
        print(key.stratum.value_counts().to_string())
        print(f"  high in slot A: {(key.high_slot == 'A').sum()}  "
              f"B: {(key.high_slot == 'B').sum()}")
        est = key.ctx_sec.sum() * 3 + 12 * len(key)
        print(f"  {len(key)} trials, ~{est/60:.1f} min of trials (+4 min overhead)")
        print(f"  -> {out_dir}")

    # Confirm the sets are disjoint
    stems = [set(pd.read_csv(out_root / f"set_{chr(ord('A')+i)}" / "answer_key.csv")
                 .query("kind == 'test'").ctx_stem) for i in range(args.n_sets)]
    overlap = set.intersection(*stems) if len(stems) > 1 else set()
    print(f"\nContexts used: {len(set.union(*stems))}  overlap between sets: {len(overlap)}")


if __name__ == "__main__":
    main()
