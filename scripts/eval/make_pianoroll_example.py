r"""
scripts/eval/make_pianoroll_example.py

Figure 3 of the paper: one context and its
high- and low-similarity outputs from one draw, as three piano rolls on a shared time axis,
the first --bars bars.

The context is black. In the outputs, red marks the right-hand melody notes that lie in a
three-interval pattern also found in the context's right-hand melody; every other note is
grey. The files carry no hand, so the right hand is taken as the notes from middle C up
(MIDI 60), and its melody as the highest of them at each onset. This keeps the melody from
falling into the bass where the right hand rests, as the skyline of the Melody measure
does. The patterns are taken over the bars shown. Each output's Melody score, from the
Melody measure over the whole section, is printed beside it.

For display only, a note lasts at most until the next onset in its hand and at most one
bar: a few context notes are held for several bars in the source.

Outputs start at time 0 and were cut at the context's length (correct_generated_output.py),
so the three rolls share one axis, in 4/4 bars of 50 ticks per beat.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\make_pianoroll_example.py --ctx 00410_s1_B_9-16 --draw 5 --bars 6
Writes paper/figures/pianoroll_example.pdf and .png (or --out).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from similarity import midi_to_compound  # noqa: E402
from symbolic_measures import TPB, interval_3gram_sim, pitches  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
GEN = {0: Path("outputs/generated/m1_triplet"), "rest": Path("outputs/generated/m1_resample")}
OUT = Path("paper/figures/pianoroll_example")
INK, OTHER, RED = "#222222", "#A8A8A8", "#C0392B"
SPLIT = 60                                                  # middle C: right hand from here up
GAP = 6                                                     # ticks left blank at the end of each drawn note
BAR_TICKS = 4 * TPB


def load_npy(path: Path) -> np.ndarray:
    t = np.load(str(path)).astype(np.int64)
    return t[np.argsort(t[:, 0], kind="stable")]


def output_path(ctx: str, label: str, draw: int) -> Path:
    if draw == 0:
        return GEN[0] / f"{ctx}_m1_{label}_generated.mid"
    return GEN["rest"] / f"{ctx}_m1_{label}_s{draw}_generated.mid"


def window(t: np.ndarray, start: int, bars: int) -> np.ndarray:
    """Notes of t that start within the first `bars` bars, times shifted to start at 0."""
    t = t.copy()
    t[:, 0] -= start
    return t[t[:, 0] < bars * BAR_TICKS]


def right_melody(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(onsets, pitches) of the highest right-hand note at each right-hand onset."""
    p = pitches(t)
    rh = t[p >= SPLIT]
    if len(rh) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    rp = pitches(rh)
    on = np.unique(rh[:, 0])
    return on, np.array([rp[rh[:, 0] == o].max() for o in on])


def shared_melody(ctx: np.ndarray, t: np.ndarray) -> set[tuple[int, int]]:
    """(onset, pitch) of right-hand melody notes of t in a three-interval pattern of the context's."""
    _, cm = right_melody(ctx)
    iv = np.diff(cm)
    grams = {tuple(iv[i:i + 3]) for i in range(len(iv) - 2)}
    on, mel = right_melody(t)
    iv = np.diff(mel)
    hit = set()
    for i in range(len(iv) - 2):
        if tuple(iv[i:i + 3]) in grams:
            hit.update(zip(on[i:i + 4].tolist(), mel[i:i + 4].tolist()))
    return hit


def display_ends(t: np.ndarray) -> np.ndarray:
    """End of each note for drawing: at most the next onset in its hand, at most one bar."""
    p = pitches(t)
    ends = t[:, 0] + np.minimum(t[:, 1], BAR_TICKS)
    for hand in (p >= SPLIT, p < SPLIT):
        on = np.unique(t[hand, 0])
        for i in np.flatnonzero(hand):
            later = on[on > t[i, 0]]
            if len(later):
                ends[i] = min(ends[i], later[0])
    return ends


def draw_roll(ax, t: np.ndarray, hits: set[tuple[int, int]] | None) -> None:
    for (o, _, *_), e, p in zip(t, display_ends(t), pitches(t)):
        red = hits is not None and (int(o), int(p)) in hits
        color = INK if hits is None else RED if red else OTHER
        # a small gap at the end of each note keeps repeated notes apart
        ax.add_patch(Rectangle((o / TPB, p - 0.4), max(e - o - GAP, 3) / TPB, 0.8,
                               facecolor=color, edgecolor="none", zorder=3 if red else 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--draw", type=int, required=True)
    ap.add_argument("--bars", type=int, default=6)
    ap.add_argument("--out", default=str(OUT), help="output path without extension")
    args = ap.parse_args()
    full_ctx = load_npy(SEC / f"{args.ctx}.npy")
    ctx = window(full_ctx, int(full_ctx[0, 0]), args.bars)
    rows = [("Context", ctx, None)]
    for lab in ("high", "low"):
        full = midi_to_compound(output_path(args.ctx, lab, args.draw))
        t = window(full, 0, args.bars)
        hits = shared_melody(ctx, t)
        score = interval_3gram_sim(full_ctx, full)
        print(f"{lab}: Melody {score:.4f} (whole section); {len(hits)} red notes in {args.bars} bars")
        rows.append((f"{lab.capitalize()} similarity\nMelody {score:.3f}", t, hits))
    end = args.bars * 4
    lo = min(pitches(t).min() for _, t, _ in rows) - 2
    hi = max(pitches(t).max() for _, t, _ in rows) + 2
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 3.2), sharex=True, sharey=True)
    for ax, (name, t, hits) in zip(axes, rows):
        draw_roll(ax, t, hits)
        ax.set_xlim(0, end)
        ax.set_ylim(lo, hi)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", labelpad=6)
        ax.set_yticks([])
        for b in np.arange(0, end + 0.01, 4):
            ax.axvline(b, color="#E6E6E6", linewidth=0.5, zorder=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("Bar")
    axes[-1].set_xticks(np.arange(0, end, 4) + 2)
    axes[-1].set_xticklabels([str(i + 1) for i in range(args.bars)])
    axes[-1].tick_params(axis="x", length=0)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in (INK, RED, OTHER)]
    fig.legend(handles, ["Context note", "Melody note in a pattern of the context's melody", "Other note"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in (".pdf", ".png"):
        fig.savefig(Path(args.out).with_suffix(ext), dpi=200, bbox_inches="tight")
    print(f"-> {args.out}.pdf, .png")


if __name__ == "__main__":
    main()
