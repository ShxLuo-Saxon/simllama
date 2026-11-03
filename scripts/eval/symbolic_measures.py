r"""
scripts/eval/symbolic_measures.py

Similarity measures between two symbolic passages, for comparing against
compound_sim. Shared by eval_listening_metrics.py and eval_copy_rate.py.

Input is the (N, 6) compound-token array that eval_triplet.midi_to_compound
returns: onset, duration, octave, pitch class, channel, velocity. Time is in
ticks at 50 ticks per beat (120 BPM, 1 tick = 10 ms), so a beat is 50 ticks and
a sixteenth note is 12.5 ticks.

Every measure returns a score in [0, 1]; higher means more similar. All are
transposition-invariant, like compound_sim.

compound_sim components (as published, imported unchanged)
    chroma, duration, register, density

Variants of them
    duration_beats   the duration histogram with bins rescaled to 50 ticks per
                     beat. The published bins were written for 480 ticks per
                     beat, so 90% of notes fall in the first bin.

Order-sensitive measures (compound_sim has none)
    The melody is the highest note at each onset, a standard skyline reduction.
    contour          1 - normalised edit distance between up/down/same strings
    interval_hist    overlap of melodic-interval histograms (-12..+12)
    interval_3gram   Jaccard overlap of melodic 3-interval patterns
    interval_dtw     DTW alignment cost of folded melodic intervals
    ioi_hist         overlap of inter-onset-interval histograms, in sixteenths
    ioi_mallows      Mallows (earth mover's) distance between IOI distributions
    onset_grid       cosine of onset positions within a 4-beat cycle, in sixteenths
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import (  # noqa: E402
    chroma_sim, density_sim, duration_sim, register_sim)

TPB = 50                    # ticks per beat
SIXTEENTH = TPB / 4         # 12.5 ticks
MAX_CONTOUR = 64            # compare at most this many melody steps


def pitches(t: np.ndarray) -> np.ndarray:
    return t[:, 2] * 12 + t[:, 3]


def melody(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Skyline melody: (onsets, pitches), the highest note at each onset."""
    if len(t) == 0:
        return np.zeros(0), np.zeros(0)
    p = pitches(t)
    onsets = np.unique(t[:, 0])
    top = np.array([p[t[:, 0] == o].max() for o in onsets])
    return onsets, top


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins) -> float:
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    if ha.sum() == 0 or hb.sum() == 0:
        return 0.0
    return float(1.0 - 0.5 * np.abs(ha / ha.sum() - hb / hb.sum()).sum())


def duration_beats_sim(a: np.ndarray, b: np.ndarray) -> float:
    """duration_sim with its 8 bins rescaled from 480 to 50 ticks per beat.

    Published edges 60, 120, ... at 480 tpb are a 32nd, 16th, 8th, quarter...
    Scaled by 50/480 they keep that meaning at our tick rate.
    """
    edges = np.array([0, 60, 120, 240, 480, 960, 1920, 3840, np.inf]) * TPB / 480
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return _hist_overlap(a[:, 1].astype(float), b[:, 1].astype(float), edges)


def _edit_distance(x: str, y: str) -> int:
    prev = list(range(len(y) + 1))
    for i, cx in enumerate(x, 1):
        cur = [i]
        for j, cy in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (cx != cy)))
        prev = cur
    return prev[-1]


def contour_sim(a: np.ndarray, b: np.ndarray) -> float:
    def s(t):
        _, m = melody(t)
        d = np.sign(np.diff(m))[:MAX_CONTOUR]
        return "".join("u" if v > 0 else "d" if v < 0 else "s" for v in d)
    x, y = s(a), s(b)
    if not x or not y:
        return 0.0
    return 1.0 - _edit_distance(x, y) / max(len(x), len(y))


def interval_hist_sim(a: np.ndarray, b: np.ndarray) -> float:
    ia = np.clip(np.diff(melody(a)[1]), -12, 12)
    ib = np.clip(np.diff(melody(b)[1]), -12, 12)
    return _hist_overlap(ia, ib, np.arange(-12.5, 13.5, 1.0))


def interval_ngrams(t: np.ndarray, n: int = 3) -> set[tuple]:
    iv = np.diff(melody(t)[1]).astype(int)
    return {tuple(iv[i:i + n]) for i in range(len(iv) - n + 1)}


def interval_3gram_sim(a: np.ndarray, b: np.ndarray) -> float:
    ga, gb = interval_ngrams(a), interval_ngrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def _fold(iv: np.ndarray) -> np.ndarray:
    """Fold intervals wider than an octave into [-12, +12], keeping direction."""
    return np.sign(iv) * (((np.abs(iv) - 1) % 12) + 1) * (iv != 0)


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cost = np.abs(a[i - 1] - b)
        cur = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            cur[j] = cost[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m])


def interval_dtw_sim(a: np.ndarray, b: np.ndarray) -> float:
    """DTW over folded melodic intervals, as in the ninsheetmusic-annotation
    project (structure_analysis.py): sim = exp(-cost / (len_a + len_b)).

    Unlike the n-gram measure, a near-miss interval (+2 vs +1) costs a little
    rather than breaking the match; but the alignment is global, so motifs that
    reappear in a different order are penalised.
    """
    ia = _fold(np.diff(melody(a)[1]).astype(int))
    ib = _fold(np.diff(melody(b)[1]).astype(int))
    if len(ia) == 0 or len(ib) == 0:
        return 0.0
    return float(np.exp(-_dtw(ia, ib) / (len(ia) + len(ib))))


def ioi_hist_sim(a: np.ndarray, b: np.ndarray) -> float:
    def ioi(t):
        o = np.unique(t[:, 0]) if len(t) else np.zeros(0)
        return np.clip(np.round(np.diff(o) / SIXTEENTH), 0, 16)
    return _hist_overlap(ioi(a), ioi(b), np.arange(-0.5, 17.5, 1.0))


def ioi_mallows_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Inter-onset intervals compared with the Mallows (earth mover's) distance,
    as Beltran, Liu, Mohanchandra & Toussaint (2013) did against human rhythm
    judgements. IOIs in sixteenths, capped at 16 (one bar); the distance is
    divided by that cap so the score is 1 - W / 16, in [0, 1].

    Unlike ioi_hist, a near miss (an eighth against a dotted eighth) costs
    little rather than counting as a full mismatch.
    """
    from scipy.stats import wasserstein_distance

    def ioi(t):
        o = np.unique(t[:, 0]) if len(t) else np.zeros(0)
        return np.clip(np.round(np.diff(o) / SIXTEENTH), 0, 16)
    ia, ib = ioi(a), ioi(b)
    if len(ia) == 0 or len(ib) == 0:
        return 0.0
    return float(1.0 - wasserstein_distance(ia, ib) / 16)


def onset_grid_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Where onsets fall in a 4-beat cycle, counted from the first onset.

    The bar line is not known, so this measures the rhythm pattern's shape up
    to a shift; the max over the 16 shifts makes it shift-invariant.
    """
    def grid(t):
        if len(t) == 0:
            return np.zeros(16)
        o = np.unique(t[:, 0]) - t[:, 0].min()
        h = np.bincount(np.round(o / SIXTEENTH).astype(int) % 16, minlength=16)
        return h / np.linalg.norm(h)
    ga, gb = grid(a), grid(b)
    if not ga.any() or not gb.any():
        return 0.0
    return float(max(np.dot(np.roll(ga, k), gb) for k in range(16)))


MEASURES = {
    "chroma": chroma_sim,
    "duration_v1_1": duration_sim,
    "register": register_sim,
    "density": density_sim,
    "duration_beats": duration_beats_sim,
    "contour": contour_sim,
    "interval_hist": interval_hist_sim,
    "interval_3gram": interval_3gram_sim,
    "interval_dtw": interval_dtw_sim,
    "ioi_hist": ioi_hist_sim,
    "ioi_mallows": ioi_mallows_sim,
    "onset_grid": onset_grid_sim,
}

CS_WEIGHTS = {"chroma": 0.4, "duration_v1_1": 0.3, "register": 0.1, "density": 0.1}


def all_measures(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Every measure above, plus the count-chroma variant of compound similarity and its variants."""
    m = {k: f(a, b) for k, f in MEASURES.items()}
    m["cs_v1_1"] = sum(w * m[k] for k, w in CS_WEIGHTS.items())
    m["cs_v1_1_duration_beats"] = (m["cs_v1_1"] - 0.3 * m["duration_v1_1"]
                              + 0.3 * m["duration_beats"])
    m["cs_v1_1_equal"] = np.mean([m[k] for k in CS_WEIGHTS])
    for drop in CS_WEIGHTS:            # leave one component out, renormalised
        keep = {k: w for k, w in CS_WEIGHTS.items() if k != drop}
        m[f"cs_v1_1_no_{drop}"] = sum(w * m[k] for k, w in keep.items()) / sum(keep.values())
    return m
