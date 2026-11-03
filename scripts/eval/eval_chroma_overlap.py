r"""
scripts/eval/eval_chroma_overlap.py

How close is the evaluation measure Chroma to CS-Chroma, the component that carries
four ninths of the label measure? Both compare pitch-class histograms at the best of
12 transpositions; Chroma counts each note once, CS-Chroma weights notes by duration.
Reports the Spearman correlation of the two over all labelled section pairs, and of
each other main measure with CS-Chroma for comparison.

Output: printed, and outputs/eval/chroma_overlap.txt.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\eval_chroma_overlap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from symbolic_measures import all_measures  # noqa: E402
from similarity import compound_parts  # noqa: E402
from paper_common import NAMES  # noqa: E402

from paper_common import SECTIONS as SEC  # noqa: E402
MAIN = {k: NAMES[k] for k in ("chroma", "interval_3gram", "ioi_mallows", "duration_beats")}


def load(path: Path) -> np.ndarray:
    t = np.load(str(path)).astype(np.int64)
    return t[np.argsort(t[:, 0], kind="stable")]


def main() -> None:
    pairs = pd.read_csv(SEC / "pairs.csv")
    rows = []
    for p in pairs.itertuples():
        a, b = load(SEC / p.context_file), load(SEC / p.target_file)
        m = all_measures(a, b)
        rows.append({"cs_chroma": compound_parts(a, b)["chroma"], **{k: m[k] for k in MAIN}})
    df = pd.DataFrame(rows).dropna()
    lines = [f"Pairs: {len(df)} of {len(pairs)} (pairs with every score defined)"]
    for k, name in MAIN.items():
        rho = stats.spearmanr(df.cs_chroma, df[k]).statistic
        lines.append(f"  Spearman, CS-Chroma vs {name:8s}: {rho:.3f}")
    print("\n".join(lines))
    out = Path("outputs/eval/chroma_overlap.txt")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
