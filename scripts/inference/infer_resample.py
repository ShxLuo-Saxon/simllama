r"""
scripts/inference/infer_resample.py

Repeated sampling from one trained model (M1, M2 or M3), with the prompt, note
budget, bar estimate and sampling settings of the published run
(infer_triplet.py). Only the random draw differs.

Used for the M1 stability check (draws 1-5) and for the retrained control
models (draws 0-5 for M3, 0-11 for M2), so that every ablation can average
over draws and an unconditioned model can be sampled several times for a
generate-and-select comparison.

Models
------
  m1, m3   one output per context, draw and label (sim:high / sim:mid / sim:low),
           with the context vector; M3 is given the correct label at test time
  m2       one output per context and draw; no label, no context vector
  m4       one output per context and draw; the context vector, no label

Seeds
-----
Every generation gets its own seed, derived from (draw, context index, label):

    seed = base_seed + 100000 * draw + 10 * ctx_index + label_index

(label_index 0 for M2), so a single output can be reproduced on its own, and an
interrupted job resumes by skipping files that exist. For M1, draw 0 is the
published run (seed 42, one sequential stream) and this script starts at 1.

Output naming:
  {ctx_stem}_{m1|m3}_{high|mid|low}_s{k}_generated.mid
  {ctx_stem}_{m2|m4}_s{k}_generated.mid

Usage (HPC):
    python scripts/inference/infer_resample.py --model m3 \
        --ckpt         $SCRATCH/runs/m3/best.pt \
        --config_path  configs/simllama_839M.json \
        --sections_dir $SCRATCH/data/sections \
        --pairs_csv    $SCRATCH/data/sections/pairs.csv \
        --output_dir   $SCRATCH/outputs/m3 \
        --first_sample 0 --n_samples 6
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.dataset import (
    SECTION_TOKEN_MAP,
    _bars_from_filename, _parse_num_bars, _normalize_onsets,
)
from scripts.inference.infer_simllama import (
    load_model, build_prompt, _estimate_bar_size, generate_one,
)

BUCKETS = ["sim:high", "sim:mid", "sim:low"]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def out_name(ctx_stem: str, model: str, bucket: str | None, s: int) -> str:
    if model in ("m2", "m4"):
        return f"{ctx_stem}_{model}_s{s}_generated.mid"
    return f"{ctx_stem}_{model}_{bucket.split(':')[1]}_s{s}_generated.mid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="m1", choices=["m1", "m2", "m3", "m4"])
    parser.add_argument("--ckpt", "--ckpt_m1", dest="ckpt", required=True,
                        help="best.pt of the model (--ckpt_m1 kept for the M1 job script)")
    parser.add_argument("--config_path",  required=True)
    parser.add_argument("--sections_dir", required=True)
    parser.add_argument("--pairs_csv",    required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--split",        default="test")
    parser.add_argument("--n_samples",    type=int,   default=5)
    parser.add_argument("--first_sample", type=int,   default=1,
                        help="Number of the first draw (M1's draw 0 is the published run)")
    parser.add_argument("--len_multiplier", type=float, default=1.2)
    parser.add_argument("--temperature",  type=float, default=0.8)
    parser.add_argument("--top_p",        type=float, default=0.9)
    parser.add_argument("--base_seed",    type=int,   default=42)
    parser.add_argument("--limit",        type=int,   default=0,
                        help="Only the first N contexts (pre-flight test); 0 = all")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model} from {args.ckpt} ...")
    model, tokenizer = load_model(args.ckpt, args.config_path, device,
                                  use_lora=True, lora_rank=8, lora_alpha=16)

    # Same per-context budget and bar estimate as infer_triplet.py
    pairs_df = pd.read_csv(args.pairs_csv)
    test_df = pairs_df[pairs_df["split"] == args.split].copy()
    test_df["context_stem"] = test_df["context_file"].apply(lambda f: Path(f).stem)
    test_df["target_bars"] = test_df["target_file"].apply(
        lambda f: _parse_num_bars(_bars_from_filename(f)))
    ctx_stats = test_df.groupby("context_stem").agg(
        mean_target_ntokens=("target_ntokens", "mean"),
        mean_target_bars=("target_bars", "mean"),
        context_file=("context_file", "first"),
    ).reset_index()
    if args.limit:
        ctx_stats = ctx_stats.head(args.limit)

    samples = range(args.first_sample, args.first_sample + args.n_samples)
    buckets = [None] if args.model in ("m2", "m4") else BUCKETS
    print(f"Contexts in {args.split} split: {len(ctx_stats)}  draws {list(samples)}  "
          f"labels {len(buckets)}")
    sections_dir = Path(args.sections_dir)
    written = 0

    for ci, row in ctx_stats.iterrows():
        ctx_stem = row["context_stem"]
        ctx_path = sections_dir / row["context_file"]
        if not ctx_path.exists():
            print(f"SKIP (missing npy): {ctx_path}")
            continue

        todo = [(s, bi, b) for s in samples for bi, b in enumerate(buckets)
                if not (out_dir / out_name(ctx_stem, args.model, b, s)).exists()]
        if not todo:
            continue

        ctx_tokens = np.load(str(ctx_path)).astype(np.int64)
        bar_size_est = _estimate_bar_size(ctx_tokens, _bars_from_filename(row["context_file"]))
        target_bars = max(1, round(row["mean_target_bars"]))
        budget = max(1, int(row["mean_target_ntokens"] * args.len_multiplier))
        print(f"\n== [{ci + 1}/{len(ctx_stats)}] {ctx_stem}  budget={budget}  "
              f"est_bars={target_bars}  todo={len(todo)} ==")

        proto_vec = None
        if args.model != "m2":
            ctx_only = torch.tensor([_normalize_onsets(ctx_tokens)[:509].tolist()],
                                    dtype=torch.long, device=device)
            with torch.inference_mode():
                proto_vec = model.compute_proto_vec(ctx_only, SECTION_TOKEN_MAP)
                if next(model.parameters()).dtype == torch.bfloat16:
                    proto_vec = proto_vec.to(torch.bfloat16)

        for s, bi, bucket in todo:
            out_path = out_dir / out_name(ctx_stem, args.model, bucket, s)
            set_seed(args.base_seed + 100000 * s + 10 * ci + bi)
            # M1 and M3 share the prompt format; M3 is given the correct label here
            mode = args.model if args.model in ("m2", "m4") else "m1"
            prompt_ids = build_prompt(tokenizer, ctx_tokens, bucket or "sim:high", mode).to(device)
            generated = generate_one(
                model=model, tokenizer=tokenizer, prompt_ids=prompt_ids,
                additional_token_map=SECTION_TOKEN_MAP, proto_vec=proto_vec,
                bar_size_est=bar_size_est, target_bars=target_bars,
                condition_mode=mode, max_gen_len=budget,
                temperature=args.temperature, top_p=args.top_p, device=device,
            )
            print(f"  s{s}/{bucket or 'none'}: {len(generated)} notes", end="  ")
            if generated:
                try:
                    tokenizer.compound_to_midi(generated).save(str(out_path))
                    written += 1
                    print(f"-> {out_path.name}")
                except Exception as e:                       # noqa: BLE001
                    print(f"MIDI save failed: {e}")
            else:
                print("empty output -- skipped")

    print(f"\nDone. Files written this run: {written}")


if __name__ == "__main__":
    main()
