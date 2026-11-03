r"""
scripts/eval/eval_train_overlap.py

Does a test context's theme appear in the training pieces under another title?
The split keeps arrangements of
one tune together by title and length, but a theme can recur across games.

For each of the 157 test contexts, against every section of the training pieces:
  max_melody  the highest Melody score (Jaccard overlap of three-interval patterns)
  longest_run the longest melodic run shared with any training section, in notes
A context has a near-duplicate in training when it shares a run of 8 or more notes
(the copy check's threshold, eval_copy_rate.py) with a training section.

Then the high-minus-low Melody gap (valid draws averaged per context, as Table 2)
is repeated on the contexts without a near-duplicate, with a 95% interval from
resampling whole pieces.

Usage:
    .venv\Scripts\python.exe scripts\eval\eval_train_overlap.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_copy_rate import MIN_COPY, SEC, runs, seqs  # noqa: E402
from paper_common import cluster_boot, load_generated  # noqa: E402
from symbolic_measures import interval_ngrams  # noqa: E402


def load(f: str) -> np.ndarray:
    t = np.load(str(SEC / f))
    return t[np.argsort(t[:, 0], kind="stable")]


def gap_line(name: str, gap: pd.Series, rng) -> str:
    means, _ = cluster_boot(gap, rng, 2000)
    return (f"  {name:34s} n {len(gap):3d} ({gap.index.str[:5].nunique()} pieces)   "
            f"Melody gap {gap.mean():.4f} [{np.percentile(means, 2.5):.4f}, {np.percentile(means, 97.5):.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="outputs/eval/robustness_scores.csv")
    ap.add_argument("--out_csv", default="outputs/eval/train_overlap.csv")
    ap.add_argument("--out", default="outputs/eval/train_overlap.txt")
    args = ap.parse_args()

    pairs = pd.read_csv(SEC / "pairs.csv")
    train = sorted(set(pairs[pairs.split == "train"].context_file) | set(pairs[pairs.split == "train"].target_file))
    stems = sorted(set(pd.read_csv("outputs/eval/contexts_all.csv").ctx_stem))
    tr = [(f, load(f)) for f in train]
    tr_grams = [interval_ngrams(t) for _, t in tr]
    tr_iv = [seqs(t)[0] for _, t in tr]

    rows = []
    for s in stems:
        ctx = load(f"{s}.npy")
        g, iv = interval_ngrams(ctx), seqs(ctx)[0]
        mel = [len(g & h) / len(g | h) if g and h else 0.0 for h in tr_grams]
        run = [runs(iv, x)[0] for x in tr_iv]
        i, j = int(np.argmax(mel)), int(np.argmax(run))
        rows.append(dict(ctx_stem=s, max_melody=mel[i], max_melody_section=tr[i][0],
                         longest_run=run[j], longest_run_section=tr[j][0]))
    ov = pd.DataFrame(rows).set_index("ctx_stem")
    ov.to_csv(args.out_csv)
    dup = ov.longest_run >= MIN_COPY

    df = load_generated(args.scores)
    per = df.groupby(["ctx_stem", "bucket"]).interval_3gram.mean().unstack()
    gap = (per["high"] - per["low"]).dropna()
    rng = np.random.default_rng(0)
    lines = [f"Training sections: {len(train)}; test contexts: {len(ov)}",
             f"Near-duplicate in training (shared run of {MIN_COPY}+ notes): {dup.sum()} contexts "
             f"({dup.mean():.1%}) in {ov[dup].index.str[:5].nunique()} pieces",
             f"Max Melody against training: median {ov.max_melody.median():.3f}, "
             f"share >= 0.5: {(ov.max_melody >= 0.5).mean():.1%}", "",
             gap_line("all contexts", gap, rng),
             gap_line("without a near-duplicate", gap[~gap.index.isin(ov[dup].index)], rng),
             gap_line("with a near-duplicate", gap[gap.index.isin(ov[dup].index)], rng)]
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
