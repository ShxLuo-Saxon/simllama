"""
Data preprocessing for SimLlama (prepend-conditioning version).

Outputs per file:
  [id].npy              — (N, 6) compound tokens (raw music events)
  [id]_condition.npy    — (N_bars × 2, 6) per-bar structure token prefix

Row layout of _condition.npy (2 rows per bar):
  row 2*(b-1):   [func_token_id, bar_onset_tokenizer, 0, 0, 0, 0]
  row 2*(b-1)+1: [sec_token_id,  bar_onset_tokenizer, 0, 0, 0, 0]

  col 0 = token_id            — embed_tokens uses this for supplementary_embedding lookup
  col 1 = bar_onset_tokenizer — onset in the SAME scale as MusicTokenizer's onset tokens:
                                  bar_onset_tokenizer = round(TIME_RESOLUTION * bar_seconds)
                                  where TIME_RESOLUTION = 100 ticks/sec (matches tokenizer)
                                simllama._build_position_ids moves this to
                                position_ids col 0 (onset RoPE axis, theta=199999)
  cols 2-5 = 0                — dur / octave / pitch / velocity axes all zero

  NOTE: all timing is normalised to 120 BPM.
        bar_onset_tokenizer = round(TIME_RESOLUTION * bar_offsets[b] * 60.0 / 120.0)
        (bar_offsets from music21 are in quarter-length units)
        This matches Moonbeam's pretrained distribution (ComMU fine-tune, heterogeneous
        pretrain -- author confirmed BPM normalisation is required for correct onset
        embedding alignment).  A 4/4 bar is always 200 ticks; a quarter note is 50 ticks.

Structure token IDs (factored, MUST match prepare_condition_csv.py):
  func_{label}  token_id = -(26 + func_idx)   func_idx in 0-6
  sec_{label}   token_id = -(33 + sec_idx)    sec_idx  in 0-26

No external Moonbeam imports — uses scripts.model.tokenizer.
"""

import os
import sys
import json
import string
import pathlib

import numpy as np
import pandas as pd
from tqdm import tqdm
from music21 import converter

# Add project root to path so we can import our own modules
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.model.tokenizer import MusicTokenizer

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[2]
                                 / "data" / "ninsheetmusic"))  # scores, metadata, processed/
CONDITION_CSV  = BASE_DIR / "processed" / "npy" / "condition.csv"
MID_DIR        = BASE_DIR / "processed" / "mid"
ANNOTATION_DIR = BASE_DIR / "processed" / "annotation" / "annotations"
MXL_DIR        = BASE_DIR / "processed" / "mxl"
MXL_FIX_DIR    = MXL_DIR / "fixes"

OUTPUT_DIR = BASE_DIR / "processed" / "npy" / "processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Structure vocabularies (MUST match prepare_condition_csv.py) ─────────────
# Function: 6 known labels + 's' fallback, sorted alphabetically
_FUNCTIONS = sorted(["Br", "In", "Ln", "Lp", "Ou", "St", "s"])
# ['Br', 'In', 'Ln', 'Lp', 'Ou', 'St', 's']  → 7 entries, indices 0-6

# Section: full A-Z alphabet + 's' fallback, sorted alphabetically
# Uppercase letters (65-90) sort before lowercase 's' (115) in Python
_SECTIONS = sorted(list(string.ascii_uppercase) + ["s"])
# ['A', 'B', ..., 'Z', 's']  → 27 entries, indices 0-26

FUNC_VOCAB = {lbl: i for i, lbl in enumerate(_FUNCTIONS)}   # 7 entries
SEC_VOCAB  = {lbl: i for i, lbl in enumerate(_SECTIONS)}    # 27 entries

FUNC_TOKEN_START = -26   # func token IDs: -26, -27, ..., -32
SEC_TOKEN_START  = -33   # sec  token IDs: -33, -34, ..., -59


def get_bar_offsets(mxl_path: pathlib.Path) -> dict:
    """
    Use music21 to get the exact absolute offset (in quarter lengths) of every bar.
    Handles anacrusis (pickup bars) and time-signature changes transparently.

    Returns dict: bar_num (1-indexed) → quarter-length offset from start.
    Also includes a dummy entry at (N_bars + 1) whose value is the end of the last bar,
    so callers can compute bar durations via bar_offsets[b+1] - bar_offsets[b].
    """
    offsets = {}
    try:
        score    = converter.parse(str(mxl_path))
        part     = score.parts[0]
        measures = part.getElementsByClass("Measure")

        for i, m in enumerate(measures, start=1):
            # Use sequential index `i` instead of `m.number` because pickup bars are
            # sometimes labelled Measure "0" in MXL files, while Gemini annotations
            # always count the first physical bar as Bar 1.
            offsets[i] = float(m.offset)

        # Dummy end entry — allows computing the last bar's tick span.
        last_m = measures[-1]
        offsets[len(measures) + 1] = float(last_m.offset + last_m.duration.quarterLength)

    except Exception as e:
        print(f"    Warning: Failed to parse MXL {mxl_path.name}: {e}")

    return offsets


def get_structure_for_bar(bar_num: int, sections: list) -> tuple:
    """
    Return (func_lbl, sec_lbl) active at bar_num (1-indexed, inclusive range).

    Searches the annotation 'Section' array.  Each entry has:
        {"BarRange": [start, end], "Function": "...", "Section": "..."}
    Falls back to 's' for both levels if bar_num is outside all listed ranges.
    """
    func_lbl = "s"
    sec_lbl  = "s"
    for entry in sections:
        bar_start, bar_end = entry["BarRange"]
        if bar_start <= bar_num <= bar_end:
            func_lbl = entry.get("Function", "s")
            sec_lbl  = entry.get("Section",  "s")
            break
    return func_lbl, sec_lbl


TIME_RESOLUTION = 100  # ticks per second — must match MusicTokenizer.midi_to_compound default


NORMALIZED_BPM = 120.0   # all timing normalised to 120 BPM to match Moonbeam pretrain distribution


def build_condition_prefix(annotation: dict, bar_offsets: dict) -> np.ndarray:
    """
    Build the (N_bars × 2, 6) per-bar structure token array.

    For each bar b in [1, N_bars]:
        row 2*(b-1):   [func_token_id, bar_onset_tokenizer, 0, 0, 0, 0]
        row 2*(b-1)+1: [sec_token_id,  bar_onset_tokenizer, 0, 0, 0, 0]

    bar_onset_tokenizer = round(TIME_RESOLUTION * bar_offsets[b] * 60.0 / 120.0)
        — normalised to 120 BPM; a 4/4 bar = 200 ticks, a quarter note = 50 ticks.
        Must match the calibrate_to_default_tempo=True scale used in midi_to_compound.

    N_bars is derived from bar_offsets (which includes a dummy end-of-piece entry at N+1).

    Returns np.ndarray shape (N_bars * 2, 6), dtype=np.int32
    """
    N_bars   = len(bar_offsets) - 1   # subtract dummy end entry
    sections = annotation.get("Section", [])
    rows     = []

    seconds_per_beat = 60.0 / NORMALIZED_BPM   # 0.5 s/beat at 120 BPM

    for b in range(1, N_bars + 1):
        # Convert quarter-length offset → normalised seconds → tokenizer ticks
        bar_onset_tick = round(TIME_RESOLUTION * bar_offsets.get(b, 0.0) * seconds_per_beat)

        func_lbl, sec_lbl = get_structure_for_bar(b, sections)

        # Fall back to 's' for any label not in vocabulary
        func_idx = FUNC_VOCAB.get(func_lbl, FUNC_VOCAB["s"])
        sec_idx  = SEC_VOCAB.get(sec_lbl,  SEC_VOCAB["s"])

        func_token_id = FUNC_TOKEN_START - func_idx   # -18 to -24
        sec_token_id  = SEC_TOKEN_START  - sec_idx    # -25 to -51

        rows.append([func_token_id, bar_onset_tick, 0, 0, 0, 0])
        rows.append([sec_token_id,  bar_onset_tick, 0, 0, 0, 0])

    return np.array(rows, dtype=np.int32)


def main():
    print(f"Reading condition CSV: {CONDITION_CSV}")
    df = pd.read_csv(CONDITION_CSV)

    tokenizer = MusicTokenizer()

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        file_id  = str(row["id"]).zfill(5)

        main_mid = MID_DIR        / f"{file_id}.mid"
        ann_file = ANNOTATION_DIR / f"{file_id}.json"

        if not main_mid.exists():
            print(f"  Skipping {file_id}: MIDI file missing.")
            continue
        if not ann_file.exists():
            print(f"  Skipping {file_id}: annotation JSON missing.")
            continue

        # ── 1. Convert MIDI → compound tokens (120 BPM normalised), save [id].npy ─
        compounds = np.array(MusicTokenizer.midi_to_compound(str(main_mid),
                                                              calibrate_to_default_tempo=True))
        # Drop notes with duration > 1023 (tokenizer vocab max).
        # Root cause: MuseScore converts grace/ornament notes into notes with
        # near-infinite duration due to a conversion artefact. These notes also
        # have incorrect onset alignment, making them spurious chord notes.
        # Dropping is cleaner than clamping — they carry no musical information.
        MAX_DURATION = 1023
        n_before = len(compounds)
        compounds = compounds[compounds[:, 1] <= MAX_DURATION]
        n_dropped = n_before - len(compounds)
        if n_dropped:
            print(f"  {file_id}: dropped {n_dropped} ornament note(s) with duration > {MAX_DURATION} ticks")
        npy_path  = OUTPUT_DIR / f"{file_id}.npy"
        np.save(str(npy_path), compounds)

        # ── 2. Load annotation JSON ───────────────────────────────────────────
        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        except Exception as e:
            print(f"  Skipping {file_id}: cannot parse annotation ({e}).")
            continue

        # ── 3. Get exact bar offsets from MXL (falls back if MXL missing) ────
        mxl_path = MXL_FIX_DIR / f"{file_id}.mxl"
        if not mxl_path.exists():
            mxl_path = MXL_DIR / f"{file_id}.mxl"

        if mxl_path.exists():
            bar_offsets = get_bar_offsets(mxl_path)
        else:
            print(f"  Warning: No MXL for {file_id}. Falling back to 4/4 uniform bar offsets.")
            # Simple fallback: assume 4/4 time, 4 quarter-note beats per bar
            N_fallback = int(row.get("num_measures", 32))
            bar_offsets = {b: (b - 1) * 4.0 for b in range(1, N_fallback + 2)}

        if not bar_offsets:
            print(f"  Skipping {file_id}: bar_offsets is empty.")
            continue

        # ── 4. Build and save _condition.npy ─────────────────────────────────
        condition_prefix = build_condition_prefix(annotation, bar_offsets)
        cond_npy_path    = OUTPUT_DIR / f"{file_id}_condition.npy"
        np.save(str(cond_npy_path), condition_prefix)

        N_bars = len(bar_offsets) - 1
        df.at[index, "num_measures"] = N_bars
        print(f"  {file_id}: {N_bars} bars -> condition shape {condition_prefix.shape}")

    # Write back updated num_measures
    df.to_csv(CONDITION_CSV, index=False)
    print(f"\nUpdated {CONDITION_CSV} with true bar counts.")
    print("Data preprocessing complete — _condition.npy files generated.")


if __name__ == "__main__":
    main()
