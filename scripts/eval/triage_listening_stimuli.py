r"""
scripts/eval/triage_listening_stimuli.py

**Measurement + filtering stage** for the listening test. Reads a CORRECTED
generation directory, measures every context, applies the filters, assigns strata,
and writes one table. The stimulus builder then does selection and rendering only.

Reads corrected output
----------------------
Point --gen_dir at a directory produced by scripts/eval/correct_generated_output.py.
That script applies the 1-tick duration repair, truncation to the context span, and
note-off clamping in one place. **No correction is applied here.**

Filters, in the order applied
-----------------------------
Measurement is cheapest to do once, so everything is measured first; the filters
then run cheapest-and-biggest-cut first.

  1. LENGTH (budget control only)
     context 12-17 s -- the 8-bar frame. 8 bars is the modal section length in the
     corpus (46% of main-body sections). Shorter gives too little to judge against;
     longer breaks the session budget.

  2. QUALITY (is the audio usable?)
     - both outputs present with >= MIN_NOTES notes
     - coverage    >= --min_coverage : output spans most of the context. One that
                                       stops early is heard as *broken*, not as
                                       *dissimilar*.
     - pitch_ratio >= --min_pitch_ratio : unique pitches relative to the context.
                                       Catches generations stuck on 1-2 pitches,
                                       which the metric misreads as low similarity.
     - stuck melody: the melody (top note per onset) uses <= --max_stuck_notes
                       distinct pitches while the reference melody uses >=
                       --min_ref_notes. Checked on the full clip AND the first
                       --stuck_window_sec, because Test 2 plays only 10 s.
                       Catches outputs that loop 2-3 notes, which pass the
                       pitch-ratio filter when the accompaniment still moves.
                       Thresholds set from the 70-context pool: at
                       <= 2 notes it missed 3 of 4 clips flagged by ear; at
                       <= 4 it removed 9 of 30 drawn trials. <= 3 removed the 4
                       flagged clips plus 1 more stuck in its first 10 s.
     All of these take the WORSE of the two outputs, so they are condition-blind
     and cannot preferentially remove sim:high or sim:low.

  3. STRATA (sign only)
     - P-pos : positive delta. Q1 is tested on these.
     - N-neg : negative delta. Metric-divergence probe, reported separately -- the
               model was asked for high similarity but the metric says it produced
               something less similar.

     NO dead band and NO median split. Both were dropped once the design settled on
     two questions:
       * the median split existed to spread delta for the delta-slope test, which
         Q2b (perceived similarity vs cs(ctx, chosen)) supersedes -- an ordinal
         outcome carries more information per trial than a binary one;
       * the dead band excluded |delta| < 0.015 on the grounds that the metric's
         ordering is noise there. But Q1 scores against the LABEL, not delta, and
         excluding near-zero cases systematically removes the contexts where the
         metric is least informative -- i.e. filters out its failures, biasing the
         sample toward metric agreement. That is exactly what Q2b exists to test.

     Cost accepted: if the metric IS right at small delta, ~2-3 trials per
     participant are genuine coin flips that dilute the Q1 rate. Delta is recorded
     per trial so a near-zero subset can be inspected post hoc.

delta = cs(ctx, gen_high) - cs(ctx, gen_low), signed.

Usage:
    .venv\Scripts\python.exe scripts\eval\triage_listening_stimuli.py `
        --gen_dir      outputs\generated\m1_listening_candidates `
        --sections_dir "data\sections" `
        --out_dir      <build folder>

The table is written INTO the build directory, so a build is self-contained and its
provenance cannot drift if triage is re-run later with different thresholds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.eval.eval_triplet import midi_to_compound, compound_sim
from scripts.eval.paper_common import (MAX_STUCK_NOTES, MIN_CONTEXT_MELODY_NOTES, MIN_COVERAGE,  # noqa: E402
                                       MIN_NOTES, MIN_PITCH_RATIO)
TICKS_PER_SEC = 100.0     # 120 BPM, ticks_per_beat=50 -> 1 tick = 0.01 s


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


def load_gen(path: Path) -> np.ndarray | None:
    """Read a corrected generated MIDI. No truncation -- already applied upstream."""
    if not path.exists():
        return None
    arr = midi_to_compound(path)
    if len(arr) > 0:
        arr = _normalize_onsets(arr)
    return arr if len(arr) >= MIN_NOTES else None


def melody_notes(a: np.ndarray, cut: int | None = None) -> int:
    """Distinct pitches in the melody line: the highest note at each onset.

    Counting all pitches misses a stuck clip whose accompaniment still moves;
    the melody is what a listener follows, and what loops audibly.
    """
    if cut is not None:
        a = a[a[:, 0] < cut]
    if len(a) == 0:
        return 0
    top: dict[int, int] = {}
    for t, p in zip(a[:, 0].tolist(), (a[:, 2] * 12 + a[:, 3]).tolist()):
        top[t] = max(top.get(t, -1), p)
    return len(set(top.values()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True,
                   help="CORRECTED generation dir (correct_generated_output.py)")
    p.add_argument("--sections_dir", required=True)
    p.add_argument("--out_dir", default="listening_test/stimuli/build",
                   help="Build directory. The triage table is written here so "
                        "each build is self-contained; the stimulus builder "
                        "reads it from the same place.")
    p.add_argument("--tag", default="m1",
                   help="Filename fragment for the generation condition")
    # 1. length
    p.add_argument("--min_ctx_sec", type=float, default=12.0)
    p.add_argument("--max_ctx_sec", type=float, default=17.0)
    # 2. quality
    p.add_argument("--min_coverage", type=float, default=MIN_COVERAGE,
                   help="Output must span this fraction of the context")
    p.add_argument("--max_stuck_notes", type=int, default=MAX_STUCK_NOTES,
                   help="Stuck melody: an output whose melody uses this many "
                        "distinct pitches or fewer...")
    p.add_argument("--min_ref_notes", type=int, default=MIN_CONTEXT_MELODY_NOTES,
                   help="...fails only if its reference melody uses at least "
                        "this many. A plain reference is not a flaw to copy.")
    p.add_argument("--stuck_window_sec", type=float, default=10.0,
                   help="Also check the first N s, since Test 2 plays only that")
    p.add_argument("--min_pitch_ratio", type=float, default=MIN_PITCH_RATIO,
                   help="Unique pitches relative to the context (worse of the two). "
                        "THRESHOLD UNDER REVIEW.")
    # 3. strata
    p.add_argument("--dead_band", type=float, default=0.0,
                   help="Drop |delta| below this. DEFAULT 0 (disabled): excluding "
                        "near-zero cases would remove the contexts where the metric "
                        "is least informative, biasing toward metric agreement. "
                        "Retained only for sensitivity analysis.")
    args = p.parse_args()

    gen_dir = Path(args.gen_dir)
    sec_dir = Path(args.sections_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate from the generated files, not from _context.mid: the triplet
    # directory has no context copies, only generated output.
    suffix = f"_{args.tag}_high_generated.mid"
    stems = sorted({f.name[: -len(suffix)]
                    for f in gen_dir.glob(f"*{suffix}")})
    print(f"Source: {gen_dir}  ({len(stems)} contexts)\n")

    # ── Measure everything once ───────────────────────────────────
    rows = []
    for stem in stems:
        ctx_npy = sec_dir / f"{stem}.npy"
        if not ctx_npy.exists():
            continue
        ctx = _normalize_onsets(np.load(str(ctx_npy)).astype(np.int64))
        if len(ctx) < MIN_NOTES:
            continue
        ctx_span = int(ctx[-1, 0])
        if ctx_span <= 0:
            continue

        hi = load_gen(gen_dir / f"{stem}_{args.tag}_high_generated.mid")
        lo = load_gen(gen_dir / f"{stem}_{args.tag}_low_generated.mid")
        if hi is None or lo is None:
            rows.append({"ctx_stem": stem, "ctx_sec": ctx_span / TICKS_PER_SEC,
                         "usable": False, "reason": "output missing or < 10 notes"})
            continue

        ctx_pitch = len(set((ctx[:, 2] * 12 + ctx[:, 3]).tolist()))
        win = int(args.stuck_window_sec * TICKS_PER_SEC)
        rows.append({
            "ctx_stem":    stem,
            "ctx_sec":     ctx_span / TICKS_PER_SEC,
            "ctx_span":    ctx_span,
            "ctx_notes":   len(ctx),
            "hi_notes":    len(hi),
            "lo_notes":    len(lo),
            "cs_high":     compound_sim(ctx, hi),
            "cs_low":      compound_sim(ctx, lo),
            "coverage":    min(hi[-1, 0], lo[-1, 0]) / ctx_span,
            "pitch_ratio": min(len(set((g[:, 2] * 12 + g[:, 3]).tolist())) / ctx_pitch
                               for g in (hi, lo)) if ctx_pitch else 0.0,
            # worse of the two outputs; each window compared with the SAME
            # window of the reference
            "mel_notes":      min(melody_notes(g) for g in (hi, lo)),
            "ref_mel_notes":  melody_notes(ctx),
            "mel_notes_win":  min(melody_notes(g, win) for g in (hi, lo)),
            "ref_mel_notes_win": melody_notes(ctx, win),
            "usable":      True,
            "reason":      "",
        })

    df = pd.DataFrame(rows)
    df["delta"] = df.cs_high - df.cs_low
    n0 = len(df)

    # ── Apply filters, reporting the funnel ───────────────────────
    print("=== FUNNEL ===")
    print(f"{'step':52s} {'pool':>5s} {'lost':>5s}")
    print(f"{'0. contexts with generated output':52s} {n0:>5d}")

    df = df[df.usable].copy()
    print(f"{'1. both outputs present, >= 10 notes':52s} {len(df):>5d} {n0 - len(df):>5d}")

    n = len(df)
    df = df[df.ctx_sec.between(args.min_ctx_sec, args.max_ctx_sec)]
    print(f"{f'2. LENGTH  context {args.min_ctx_sec:g}-{args.max_ctx_sec:g} s':52s} "
          f"{len(df):>5d} {n - len(df):>5d}")

    n = len(df)
    df = df[df.coverage >= args.min_coverage]
    print(f"{f'3. QUALITY coverage >= {args.min_coverage}':52s} {len(df):>5d} {n - len(df):>5d}")

    n = len(df)
    df = df[df.pitch_ratio >= args.min_pitch_ratio]
    print(f"{f'4. QUALITY pitch ratio >= {args.min_pitch_ratio}':52s} {len(df):>5d} {n - len(df):>5d}")

    n = len(df)
    k, r = args.max_stuck_notes, args.min_ref_notes
    stuck = (((df.mel_notes <= k) & (df.ref_mel_notes >= r))
             | ((df.mel_notes_win <= k) & (df.ref_mel_notes_win >= r)))
    df = df[~stuck]
    print(f"{f'5. QUALITY stuck melody (<= {args.max_stuck_notes} notes, ref >= {args.min_ref_notes})':52s} "
          f"{len(df):>5d} {n - len(df):>5d}")

    n = len(df)
    if args.dead_band > 0:
        df = df[df.delta.abs() >= args.dead_band].copy()
        print(f"{f'6. dead band |delta| >= {args.dead_band} (non-default)':52s} "
              f"{len(df):>5d} {n - len(df):>5d}")
    else:
        df = df.copy()

    if df.empty:
        raise SystemExit("No contexts survive the filters.")

    # ── Strata: sign only ─────────────────────────────────────────
    pos = df[df.delta > 0].copy().sort_values("delta")
    neg = df[df.delta < 0].copy().sort_values("delta")
    pos["stratum"] = "P-pos"
    neg["stratum"] = "N-neg"
    df = pd.concat([pos, neg]).reset_index(drop=True)

    print("\n=== STRATA (sign only) ===")
    g = df.groupby("stratum", observed=True).agg(
        n=("delta", "size"), dmin=("delta", "min"),
        dmed=("delta", "median"), dmax=("delta", "max")).round(4)
    print(g.reindex(["N-neg", "P-pos"]).to_string())

    # Frame-level effect, for the record (robustness check, not a reported result)
    w = wilcoxon(df.delta, alternative="greater")
    print(f"\nFrame delta: fp={(df.delta > 0).mean():.3f}  "
          f"median={df.delta.median():+.4f}  p={w.pvalue:.4f}")
    print("  (robustness check -- not for the results table)")

    out_csv = out_dir / f"triage_{args.tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df)} contexts)")
    print("Next: build_listening_stimuli.py selects and renders from this table.")


if __name__ == "__main__":
    main()
