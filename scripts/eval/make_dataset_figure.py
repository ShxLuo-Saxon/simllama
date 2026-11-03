"""
scripts/eval/make_dataset_figure.py

Figure 4 of the paper: the distribution of compound similarity (CS) over the
3,304 training pairs, with the two thresholds that make the three labels.

Output: paper/figures/dataset_dist.{pdf,png}

Usage:
    .venv/Scripts/python.exe scripts/eval/make_dataset_figure.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import PAIRS  # noqa: E402
OUT = ROOT / "paper/figures/dataset_dist"

T_LOW, T_HIGH = 0.603, 0.778                     # tertiles, section 4
C_LOW, C_MID, C_HIGH = "#c0392b", "#888888", "#2471a3"
INK = "#1f2328"

df = pd.read_csv(PAIRS)
vals = df["compound_ks"]
counts = {b: int((df.sim_bucket == f"sim:{b}").sum()) for b in ("low", "mid", "high")}

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(4.2, 2.2))
n, edges, patches = ax.hist(vals, bins=40, color=C_MID, linewidth=0.3, edgecolor="white")
for patch, left in zip(patches, edges[:-1]):
    if left < T_LOW:
        patch.set_facecolor(C_LOW)
    elif left >= T_HIGH:
        patch.set_facecolor(C_HIGH)
for t in (T_LOW, T_HIGH):
    ax.axvline(t, color="#333333", linewidth=0.8, linestyle="--")
y_top = ax.get_ylim()[1]
for x, name, color in (((0.41 + T_LOW) / 2, "low", C_LOW), ((T_LOW + T_HIGH) / 2, "mid", "#555555"),
                       ((T_HIGH + 1.0) / 2, "high", C_HIGH)):
    ax.text(x, y_top * 0.95, f"sim:{name}\n{counts[name]:,} pairs", ha="center", va="top",
            fontsize=7, color=color)
ax.set_xlabel("CS between context and target", color=INK)
ax.set_ylabel("Pairs", color=INK)
ax.set_xlim(0.38, 1.02)
ax.yaxis.grid(True, linewidth=0.3, color="#d0d7de")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300)
print(f"-> {OUT}.pdf, {OUT}.png; counts {counts}")
