r"""
scripts/eval/plot_annotation_schemes.py

Figure 1 of the paper: one piece annotated in both schemes. The AI marks one
level of lettered sections; each human marks nested levels (L1 finest), and a
letter is comparable only within its level.

Reads data/annotations/gemini_prompt_v1/<id>.json and
data/annotations/human/annotator{1,2}/<id>.csv; bars count from the first
bar of each annotation, as in eval_human_validation.py.

Output: paper/figures/annotation_schemes.{pdf,png}

Usage:
    .venv\Scripts\python.exe scripts\eval\plot_annotation_schemes.py [--piece 00005]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_human_validation import AI_DIR, HUM, human_levels  # noqa: E402

OUT = Path("paper/figures/annotation_schemes")
AI_FILL, HUMAN_FILL, EDGE, INK = "#d9d9d9", "#ffffff", "#404040", "#1a1a1a"


def rows(piece: str):
    ai = json.load(open(AI_DIR / f"{piece}.json", encoding="utf-8"))["Section"]
    a0 = min(s["BarRange"][0] for s in ai)
    out = [("AI", [(s["BarRange"][0] - a0, s["BarRange"][1] - a0 + 1, s["Section"]) for s in ai], AI_FILL)]
    for k in (1, 2):
        lv = human_levels(HUM / f"annotator{k}" / f"{piece}.csv")
        for level in sorted(lv, reverse=True):
            out.append((f"Annotator {k}, L{level}", lv[level], HUMAN_FILL))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="00005")
    args = ap.parse_args()
    data = rows(args.piece)
    end = max(b for _, segs, _ in data for _, b, _ in segs)

    plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(data) + 0.7))
    for i, (name, segs, fill) in enumerate(data):
        y = len(data) - 1 - i - (0.3 if i > 0 else 0) - (0.3 if "Annotator 2" in name else 0)
        for a, b, letter in segs:
            ax.add_patch(plt.Rectangle((a, y), b - a, 0.8, facecolor=fill, edgecolor=EDGE, linewidth=0.6))
            ax.text((a + b) / 2, y + 0.4, letter, ha="center", va="center", color=INK)
        ax.text(-0.8, y + 0.4, name, ha="right", va="center", color=INK)
    ax.set_xlim(0, end)
    ax.set_ylim(-0.7, len(data))
    ax.set_xticks(range(0, end, 8))
    ax.set_xticklabels([str(t + 1) for t in range(0, end, 8)])
    ax.set_xlabel("Bar")
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    print(f"-> {OUT}.pdf, {OUT}.png")


if __name__ == "__main__":
    main()
