r"""
scripts/eval/make_listening_tables.py

Listening tables of the supplement (paper/sections/99-supplement.md), with the
paper's measure names (paper_common.NAMES). Prints Markdown; the supplement's
numbers are copied from this output. The label comparisons are
make_label_comparison_table.py's.

  listener agreement  every paper measure against listener ratings, first 10 s
                      heard (listening_metrics.csv, eval_listening_metrics.py);
                      Holm correction over the 12 paper measures
  listening details   label effect in the listening test, other analyses: per
                      listener, per trial, excluded listeners put back, halves

The 12 paper measures are the four main ones, CS, and seven more; compound
similarity's count-chroma variant and its own variants are not paper measures.

Usage:
    .venv\Scripts\python.exe scripts\eval\make_listening_tables.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_common import NAMES, holm  # noqa: E402
from score_listening_ratings import is_survey, parse_survey, parse_wide  # noqa: E402

# code name -> paper name, main measures first
SLOT_BIAS = 7.86


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def listener_table(path: str) -> None:
    t = pd.read_csv(path)
    t = t[t.window.str.startswith("first")].set_index("measure").loc[list(NAMES)]
    t["p_holm12"] = holm(t.p)
    print("\n### Listener agreement, first 10 s, Holm over 12\n")
    print("| Measure | Rating rise per SD | 95% interval | p | Holm p | r over 60 clips |")
    print("|---|---|---|---|---|---|")
    for m, name in NAMES.items():
        r = t.loc[m]
        print(f"| {name} | {r.slope_per_sd:+.1f} | {r.ci_lo:.2f} to {r.ci_hi:.2f} | "
              f"{fmt_p(r.p)} | {fmt_p(r.p_holm12)} | {r.clip_r:.2f} |")


def listening_robustness(key_path: str, resp: str) -> None:
    key = pd.read_csv(key_path)
    src = Path(resp)
    files = sorted(src.glob("*.csv")) if src.is_dir() else [src]
    long = pd.concat([parse_survey(f) if is_survey(f) else parse_wide(f)
                      for f in files], ignore_index=True)
    w = long.pivot_table(index=["participant_id", "trial"], columns="slot",
                         values="rating").reset_index()
    d = key.merge(w, on="trial")
    d["diff"] = np.where(d.high_slot == "A", d.A - d.B, d.B - d.A)
    # slot bias removed: A is rated SLOT_BIAS points high whatever it holds
    d["diff_adj"] = d["diff"] - SLOT_BIAS * np.where(d.high_slot == "A", 1, -1)
    bad = d[(d.kind == "check") & (d["diff"] <= 0)].participant_id.unique()
    test = d[d.kind == "test"]
    valid = test[~test.participant_id.isin(bad)]
    order = {t: i for i, t in enumerate(sorted(key.trial))}

    print("\n### Label effect, other analyses (all 30 test trials)\n")
    per_l = valid.groupby("participant_id").diff_adj.mean()
    p_l = stats.wilcoxon(per_l, alternative="greater").pvalue
    print(f"per listener: {(per_l > 0).sum()} of {len(per_l)} positive, "
          f"mean {per_l.mean():+.1f}, Wilcoxon one-sided p {p_l:.4f}")
    per_t = valid.groupby("trial").diff_adj.mean()
    p_t = stats.wilcoxon(per_t, alternative="greater").pvalue
    print(f"per trial: {(per_t > 0).sum()} of {len(per_t)} positive, "
          f"mean {per_t.mean():+.1f}, Wilcoxon one-sided p {p_t:.4f}")
    all_l = test.groupby("participant_id").diff_adj.mean()
    p_a = stats.wilcoxon(all_l, alternative="greater").pvalue
    print(f"excluded put back: {len(all_l)} listeners, {(all_l > 0).sum()} positive, "
          f"mean {all_l.mean():+.1f}, Wilcoxon one-sided p {p_a:.4f}")
    v = valid.assign(half=np.where(valid.trial.map(order) < len(order) / 2, 1, 2))
    h = v.groupby(["participant_id", "half"]).diff_adj.mean().unstack()
    p_h = stats.wilcoxon(h[1], h[2]).pvalue
    print(f"halves (trials 1-16 vs 17-32): mean {h[1].mean():+.1f} vs {h[2].mean():+.1f}, "
          f"paired Wilcoxon two-sided p {p_h:.3f}")
    print(f"excluded listeners: {len(bad)}; valid: {valid.participant_id.nunique()}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--listening", default="outputs/eval/listening_metrics.csv")
    ap.add_argument("--key", default="listening_test/answer_key.csv")
    ap.add_argument("--responses", default="listening_test/ratings.csv")
    args = ap.parse_args()

    listener_table(args.listening)
    listening_robustness(args.key, args.responses)


if __name__ == "__main__":
    main()
