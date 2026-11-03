r"""
scripts/eval/similarity.py

**The single source for every similarity metric and the MIDI reader in this
project.** No other script defines its own copy; they import from here.
`test_similarity.py` checks that this module reproduces every published number.

Two compound metrics, kept separate on purpose
----------------------------------------------
Names: compound similarity (compound_similarity) and a count-chroma variant
(compound_similarity_v1_1). The paper uses only the first and calls it "compound
similarity". Legacy names stay as aliases so that published scripts run unchanged:
compound_similarity = compute_compound = compound_ks,
compound_similarity_v1_1 = compound_sim.

They share four part names but are computed differently. The published numbers
depend on each exactly as written, so neither is "fixed" here: a fix would be a
new, named variant (see symbolic_measures.py), never an edit.

compound_similarity(a, b)   (legacy: compute_compound, compound_ks)
    Built the training buckets (`compound_ks` column of pairs.csv, written by
    scripts/data/build_pairs.py) and the published AA/AB
    check (AUC 0.834). Returns a weighted mean, weights renormalised to 1:

      0.4 chroma    pitch-class histogram, each note weighted by its duration;
                    B is transposed by the key shift k (0..11) that maximises
                    the cosine with A; score = that cosine
      0.3 duration  16 log-spaced bins between EACH SECTION'S OWN shortest and
                    longest note; cosine of the two histograms. Because the bin
                    edges differ per section, bin i means a different note
                    length in A and in B: this compares the SHAPE of the
                    duration spread, not actual durations
      0.1 register  1 - |mean pitch A - mean pitch B| / 48, floored at 0
      0.1 density   1 - |notes per tick A - B| / max of the two, in [0, 1]

compound_similarity_v1_1(a, b)   (legacy: compound_sim)
    Scored all model outputs (eval_triplet.py: headline H1_ctx) and the
    listening-test stimuli. Returns a weighted SUM whose weights total 0.9:

      0.4 chroma    pitch-class histogram, each note counted once; max cosine
                    over 12 cyclic shifts
      0.3 duration  8 fixed bins with edges 0, 60, 120, 240, 480, 960, 1920,
                    3840 ticks; score = 1 - half the L1 distance. The edges are
                    for 480 ticks per beat, but the data use 50 (a beat is 50
                    ticks), so ~90% of notes fall in the first bin: in practice
                    the term compares the share of notes longer than about one
                    beat. The paper's "32nd through double whole" describes the
                    intent, not this code
      0.1 register  1 - |mean pitch A - mean pitch B| / 88, floored at 0
      0.1 density   1 - |notes per tick A - B| / max of the two, floored at 0

Time base: 50 ticks per beat at 120 BPM, so 1 tick = 10 ms.
"""
from __future__ import annotations

from pathlib import Path

import mido
import numpy as np


# ── MIDI -> compound tokens ────────────────────────────────────────────────────

def midi_to_compound(midi_path: Path) -> np.ndarray:
    """Load generated MIDI and return (N, 6) compound tokens.

    Generated MIDIs use ticks_per_beat=50 (from tokenizer.compound_to_midi with
    TIME_RESOLUTION=100).  We do NOT rescale to 480 tpb — keeping the native
    tick scale so that onset/duration values match the .npy reference files
    (which also use TIME_RESOLUTION=100).
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return np.zeros((0, 6), dtype=np.int64)

    notes: list[list[int]] = []
    for track in mid.tracks:
        abs_tick = 0
        # (channel, note) -> (onset, velocity). Velocity MUST come from the
        # note_on: a note_off's velocity is 0 (or a release value), and reading it
        # there set every note to velocity 0. compound_sim never uses velocity,
        # so the metric was unaffected -- but any array turned back into MIDI
        # rendered as silence (seen in a listening-test stimulus).
        active: dict[tuple[int, int], tuple[int, int]] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    onset, velocity = active.pop(key)
                    duration = max(1, abs_tick - onset)
                    octave = msg.note // 12
                    pitch = msg.note % 12
                    notes.append([onset, duration, octave, pitch, msg.channel % 16,
                                  min(velocity, 127)])

    if not notes:
        return np.zeros((0, 6), dtype=np.int64)
    return np.array(sorted(notes, key=lambda x: x[0]), dtype=np.int64)


# ── compound_sim ──────────────────────────────────────────────────────────────
# Weights: chroma 0.4 + duration 0.3 + register 0.1 + density 0.1 = 0.9 (intentional)

def chroma_vector(tokens: np.ndarray) -> np.ndarray:
    chroma = np.zeros(12)
    for row in tokens:
        chroma[int(row[3]) % 12] += 1
    s = chroma.sum()
    return chroma / s if s > 0 else chroma


def chroma_sim(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = chroma_vector(a), chroma_vector(b)
    na, nb = np.linalg.norm(ca), np.linalg.norm(cb)
    if na == 0 or nb == 0:
        return 0.0
    return float(max(np.dot(np.roll(ca, k), cb) / (na * nb) for k in range(12)))


def duration_sim(a: np.ndarray, b: np.ndarray) -> float:
    bins = np.array([0, 60, 120, 240, 480, 960, 1920, 3840, np.inf])
    def hist(tok):
        if len(tok) == 0:
            return np.zeros(8)
        h, _ = np.histogram(tok[:, 1].astype(float), bins=bins)
        s = h.sum()
        return h / s if s > 0 else h.astype(float)
    da, db = hist(a), hist(b)
    return float(1.0 - 0.5 * np.sum(np.abs(da - db)))


def register_sim(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    ma = float(np.mean(a[:, 2] * 12 + a[:, 3]))
    mb = float(np.mean(b[:, 2] * 12 + b[:, 3]))
    return float(max(0.0, 1.0 - abs(ma - mb) / 88.0))


def density_sim(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    da = len(a) / max(1, int(a[-1, 0] - a[0, 0]) + 1)
    db = len(b) / max(1, int(b[-1, 0] - b[0, 0]) + 1)
    return float(max(0.0, 1.0 - abs(da - db) / max(da, db, 1e-9)))


def compound_sim(a: np.ndarray, b: np.ndarray) -> float:
    return (0.4 * chroma_sim(a, b) +
            0.3 * duration_sim(a, b) +
            0.1 * register_sim(a, b) +
            0.1 * density_sim(a, b))


# ── compound_ks (training buckets, AA/AB check) ──────────────────────────────

def chroma_weighted(arr: np.ndarray, key_shift: int = 0) -> np.ndarray | None:
    chroma = np.zeros(12, dtype=float)
    for row in arr:
        pc  = (int(row[3]) + key_shift) % 12
        dur = max(float(row[1]), 0.0)
        chroma[pc] += dur
    total = chroma.sum()
    return chroma / total if total > 1e-9 else None


def duration_hist(arr: np.ndarray, n_bins: int = 16) -> np.ndarray | None:
    durs = arr[:, 1].astype(float)
    durs = durs[durs > 1e-6]
    if len(durs) < 2:
        return None
    lo, hi = durs.min(), durs.max()
    if hi <= lo:
        return None
    bins = np.logspace(np.log10(lo + 1e-6), np.log10(hi + 1e-6), n_bins + 1)
    hist, _ = np.histogram(durs, bins=bins)
    total = hist.sum()
    return hist.astype(float) / total if total > 0 else None


def mean_pitch(arr: np.ndarray) -> float:
    if len(arr) == 0:
        return float("nan")
    return float((arr[:, 2] * 12 + arr[:, 3]).mean())


def note_density(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("nan")
    span = arr[:, 0].max() - arr[:, 0].min()
    return len(arr) / span if span > 1e-6 else float("nan")


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-9 else float("nan")


def compound_parts(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """The four component scores of compound similarity (NaN when undefined).

    chroma    duration-weighted pitch-class histograms, cosine at the best key shift
    duration  16 log bins per section, cosine
    register  1 - |mean pitch difference| / 48, floored at 0
    density   1 - |note-rate difference| / larger rate
    """
    # find best key shift
    ch_a = chroma_weighted(a, 0)
    best_k, best_cos = 0, -1.0
    if ch_a is not None:
        for k in range(12):
            ch_b = chroma_weighted(b, k)
            c = cosine(ch_a, ch_b)
            if not np.isnan(c) and c > best_cos:
                best_cos, best_k = c, k
    else:
        best_cos = float("nan")

    ch_b_ks = chroma_weighted(b, best_k)
    chroma_s = cosine(ch_a, ch_b_ks)

    dur_a, dur_b = duration_hist(a), duration_hist(b)
    duration_s = cosine(dur_a, dur_b)

    mp_a, mp_b = mean_pitch(a), mean_pitch(b)
    MAX_PITCH_DIFF = 48.0
    pitch_diff_norm = min(abs(mp_a - mp_b) / MAX_PITCH_DIFF, 1.0) if not (np.isnan(mp_a) or np.isnan(mp_b)) else float("nan")
    register_s = 1.0 - pitch_diff_norm if not np.isnan(pitch_diff_norm) else float("nan")

    dn_a, dn_b = note_density(a), note_density(b)
    if not (np.isnan(dn_a) or np.isnan(dn_b)) and max(dn_a, dn_b) > 1e-9:
        density_s = float(np.clip(1.0 - abs(dn_a - dn_b) / max(dn_a, dn_b), 0.0, 1.0))
    else:
        density_s = float("nan")
    return dict(chroma=chroma_s, duration=duration_s, register=register_s,
                density=density_s)


def compute_compound(a: np.ndarray, b: np.ndarray,
                     w_ch: float = 0.4, w_du: float = 0.3,
                     w_rg: float = 0.1, w_de: float = 0.1) -> float:
    """Compound_ks: find optimal key shift then compute weighted similarity."""
    p = compound_parts(a, b)
    chroma_s, duration_s = p["chroma"], p["duration"]
    register_s, density_s = p["register"], p["density"]
    components = [
        w_ch * chroma_s   if not np.isnan(chroma_s)   else None,
        w_du * duration_s if not np.isnan(duration_s) else None,
        w_rg * register_s if not np.isnan(register_s) else None,
        w_de * density_s  if not np.isnan(density_s)  else None,
    ]
    valid = [v for v in components if v is not None]
    if not valid:
        return float("nan")
    total_w = sum([w_ch, w_du, w_rg, w_de][:len(components)])
    # renormalise by actual weight sum of non-nan components
    weights = [w_ch, w_du, w_rg, w_de]
    valid_w = sum(w for w, v in zip(weights, components) if v is not None)
    return sum(valid) / valid_w if valid_w > 1e-9 else float("nan")




# ── Canonical names ───────────────────────────────────────────────────────────
# compound_similarity built the training labels and is the paper's "compound
# similarity"; compound_similarity_v1_1 is the count-chroma variant, not used in the paper.
compound_similarity = compute_compound
compound_similarity_v1_1 = compound_sim
# Legacy alias kept for scripts written before the rename.
compound_ks = compute_compound
