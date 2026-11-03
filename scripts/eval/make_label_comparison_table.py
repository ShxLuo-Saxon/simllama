r"""
scripts/eval/make_label_comparison_table.py

Table S2 of the supplement: every comparison on all 12 measures, without and after the
quality filter, from outputs/eval/effect_sizes.csv. Prints the Markdown rows in the order
and format of paper/sections/99-supplement.md (all 157 test contexts).

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\make_label_comparison_table.py
"""
import pandas as pd

ORDER = ["Melody", "Chroma", "Rhythm", "Duration", "CS (ours)", "Melody-Contour", "Melody-Intervals",
         "Melody-Alignment", "Rhythm-Overlap", "Rhythm-Grid", "Register", "Density"]
COMPS = [("high-low", "high > low"), ("high-mid", "high > mid"), ("mid-low", "mid > low")]


def gap(r) -> str:
    return f"{r.mean_gap:+.4f} ({r.gap_lo:+.4f} to {r.gap_hi:+.4f})"


def main() -> None:
    t = pd.read_csv("outputs/eval/effect_sizes.csv")
    t["comparison"] = t.comparison.str.replace(">", "-")
    idx = t.set_index(["quality_filter", "comparison", "measure"])
    print("| Measure | Comparison | Share | Mean Gap (95% Interval) | Holm p | After the Filter: Mean Gap; Holm p |")
    print("|---|---|---:|---:|---:|---:|")
    for m in ORDER:
        for i, (key, label) in enumerate(COMPS):
            a, f = idx.loc[("none", key, m)], idx.loc[("draw", key, m)]
            print(f"| {m if i == 0 else ''} | {label} | {a.share:.1%} | {gap(a)} | {a.p_piece_holm:.3f} | "
                  f"{gap(f)}; {f.p_piece_holm:.3f} |")


if __name__ == "__main__":
    main()
