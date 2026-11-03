r"""
scripts/eval/paper_common.py

The analysis rules of the paper, defined once. Every eval script imports them from
here and copies none.
After a change here, run again every script that imports this module and compare each
file in outputs/eval/ with the one before.

  ROOT, SECTIONS, PAIRS  the repository root, the section arrays, the pair list
  NAMES, MAIN        code name -> paper name of the 12 measures, main measures first;
                     MAIN is the four main measures and CS (the "Names" table)
  MIN_NOTES          the ten-note rule: a draw counts only if every output in it has
                     at least this many notes, so that every measure can be computed
  MIN_COVERAGE, MIN_PITCH_RATIO, MAX_STUCK_NOTES, MIN_CONTEXT_MELODY_NOTES
                     the stimulus filter (listening test, section 8); an output is
                     unstable if it fails any of them
  output_flags()     the stimulus-filter quantities of one output against its context
  unstable()         the stimulus filter applied to a frame of scored outputs
  valid_draws()      the ten-note rule applied to a frame of scored outputs
  load_generated()   scored outputs with the ten-note rule and an optional filter
  holm()             Holm correction
  cluster_boot()     resampling of whole pieces: mean gap and share of positive gaps
  positive_share_boot()  resampling of whole pieces: share of positive values

A piece is the first five characters of a context stem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]                # the repository root
SECTIONS = ROOT / "data/sections"           # section arrays and pair list
PAIRS = SECTIONS / "pairs.csv"

NAMES = {"interval_3gram": "Melody", "chroma": "Chroma", "ioi_mallows": "Rhythm",
         "duration_beats": "Duration", "cs": "CS (ours)", "contour": "Melody-Contour",
         "interval_hist": "Melody-Intervals", "interval_dtw": "Melody-Alignment",
         "ioi_hist": "Rhythm-Overlap", "onset_grid": "Rhythm-Grid", "register": "Register",
         "density": "Density"}
MAIN = {k: NAMES[k] for k in ("interval_3gram", "chroma", "ioi_mallows", "duration_beats", "cs")}

MIN_NOTES = 10
MIN_COVERAGE = 0.55               # share of the context's span the output covers
MIN_PITCH_RATIO = 0.15            # distinct pitches, relative to the context's
MAX_STUCK_NOTES = 3               # a melody with this many distinct pitches or fewer is stuck ...
MIN_CONTEXT_MELODY_NOTES = 8      # ... when the context's melody has at least this many


def piece(ctx_stem: str) -> str:
    return ctx_stem[:5]


def output_flags(ctx: np.ndarray, g: np.ndarray, melody) -> dict:
    """coverage, pitch_ratio and stuck of output g (onsets from 0) against context ctx.
    melody is symbolic_measures.melody, passed in to keep this module free of MIDI code."""
    pitches = lambda t: set((t[:, 2] * 12 + t[:, 3]).tolist())  # noqa: E731
    return {"coverage": g[:, 0].max() / max(1, ctx[:, 0].max() - ctx[:, 0].min()),
            "pitch_ratio": len(pitches(g)) / max(1, len(pitches(ctx))),
            "stuck": (len(set(melody(g)[1])) <= MAX_STUCK_NOTES
                      and len(set(melody(ctx)[1])) >= MIN_CONTEXT_MELODY_NOTES)}


def unstable(df: pd.DataFrame) -> pd.Series:
    """True for each output that fails the stimulus filter."""
    return ((df.coverage < MIN_COVERAGE) | (df.pitch_ratio < MIN_PITCH_RATIO)
            | df.stuck.fillna(False).astype(bool))


def valid_draws(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the draws (ctx_stem, sample) whose outputs all have MIN_NOTES or more."""
    return df[df.groupby(["ctx_stem", "sample"]).n_notes.transform("min") >= MIN_NOTES].copy()


def load_generated(path_or_frame, quality_filter: str = "none") -> pd.DataFrame:
    """Scored outputs of valid draws. quality_filter: "none"; "draw" drops a draw if any
    of its outputs is unstable (the stimulus filter); "pair" keeps the column `bad`, so that
    a comparison drops a draw only if one of its own two outputs is unstable."""
    df = pd.read_csv(path_or_frame) if isinstance(path_or_frame, str) else path_or_frame
    df = valid_draws(df)
    df["bad"] = unstable(df)
    if quality_filter == "draw":
        df = df[~df.bad.groupby([df.ctx_stem, df["sample"]]).transform("any")]
    if quality_filter != "pair":
        df["bad"] = False
    return df


def holm(p):
    """Holm-adjusted p-values; takes an array or a Series and returns the same type."""
    if isinstance(p, pd.Series):
        return pd.Series(holm(p.to_numpy(dtype=float)), index=p.index)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def cluster_boot(gap: pd.Series, rng, n: int):
    """Resample pieces; return arrays of mean gap and positive share (ties left out).
    gap is indexed by context stem."""
    pc = gap.index.str[:5]
    groups = [gap[pc == k].to_numpy() for k in np.unique(pc)]
    means, shares = np.empty(n), np.empty(n)
    for b in range(n):
        x = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        means[b] = x.mean()
        nz = x[np.abs(x) > 1e-12]
        shares[b] = (nz > 0).mean()
    return means, shares


def positive_share_boot(x: np.ndarray, pieces: np.ndarray, rng, n: int = 2000,
                        keep: np.ndarray | None = None) -> list[float]:
    """Resample pieces; return the share of values above zero in each resample.
    keep drops values (ties) without dropping their piece from the resampling."""
    keep = np.ones(len(x), bool) if keep is None else keep
    groups = [x[(pieces == k) & keep] for k in np.unique(pieces)]
    return [np.mean(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) > 0)
            for _ in range(n)]
