"""
Generate condition.csv and indexed_tokens_dict.json for SimLlama.

Token ID layout (must match model_config JSON):
  -1  SOS
  -2  EOS
  -3  PAD
  -4  SOC
  -5  EOC
  -6 … -25  metadata tokens  (20 slots reserved)
  -26 … -32  Function tokens  (7: sorted _FUNCTIONS)
  -33 … -59  Section  tokens  (27: A-Z + 's')

Factored design — one token ID per label per level (not Cartesian product):
  func_{label}  token_id = -(26 + func_idx)   func_idx in 0-6
  sec_{label}   token_id = -(33 + sec_idx)    sec_idx  in 0-26

Total structure tokens: 7 + 27 = 34  (matches model_config num_structure_tokens)

The indexed_tokens_dict.json is passed as `additional_token_dict_path` to training.
"""

import string
import os
import pathlib
import pandas as pd
import json

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[2]
                                 / "data" / "ninsheetmusic"))  # scores, metadata, processed/
MID_DIR        = BASE_DIR / "processed" / "mid"
OUTPUT_CSV     = BASE_DIR / "processed" / "npy" / "condition.csv"
METADATA_FILE  = BASE_DIR / "metadata.xlsx"
TOKEN_DICT_OUT = BASE_DIR / "processed" / "npy" / "indexed_tokens_dict.json"

# ── Reserved token IDs ───────────────────────────────────────────────────────
# -1=SOS  -2=EOS  -3=PAD  -4=SOC  -5=EOC
METADATA_TOKEN_START    = -6    # first metadata token ID
FUNC_TOKEN_START        = -26   # first Function token ID  (must match structure_token_offset)
SEC_TOKEN_START         = -33   # first Section  token ID  (= FUNC_TOKEN_START - 7)
NUM_METADATA_VALUES_MAX = 20    # max slots reserved in supplementary_embedding (-6 to -25)

# ── Structure vocabularies ───────────────────────────────────────────────────
# Function: 6 known labels + 's' fallback, sorted alphabetically
_FUNCTIONS = sorted(["Br", "In", "Ln", "Lp", "Ou", "St", "s"])
# ['Br', 'In', 'Ln', 'Lp', 'Ou', 'St', 's']  → 7 entries, indices 0-6

# Section: full A-Z alphabet + 's' fallback, sorted alphabetically
# Uppercase letters (65-90) sort before lowercase 's' (115) in Python
_SECTIONS = sorted(list(string.ascii_uppercase) + ["s"])
# ['A', 'B', ..., 'Z', 's']  → 27 entries, indices 0-26

FUNC_VOCAB = {lbl: i for i, lbl in enumerate(_FUNCTIONS)}   # 7 entries
SEC_VOCAB  = {lbl: i for i, lbl in enumerate(_SECTIONS)}    # 27 entries

# ── Which CSV columns become metadata condition tokens ───────────────────────
# Metadata columns to include as conditioning tokens.
# To add new metadata: append a column name here — token IDs are assigned automatically
# from METADATA_TOKEN_START downward (up to NUM_METADATA_VALUES_MAX slots).
# Currently empty: time_signature was unreliable (first-occurrence labeling for mixed-meter
# pieces) and inst is always "piano" for this dataset. Slots are reserved in the model
# for future use.
METADATA_COLUMNS: list[str] = []


def main():
    # 1. Find processed MIDI files
    mid_files = list(MID_DIR.glob("*.mid"))
    file_ids  = sorted([f.stem for f in mid_files])
    print(f"Found {len(file_ids)} MIDI files")

    # 2. Load metadata Excel
    df_meta = pd.read_excel(METADATA_FILE)
    df_meta["id"] = df_meta["id"].astype(str).str.zfill(5)
    df_filtered = df_meta[df_meta["id"].isin(file_ids)].copy()

    # 3. Build per-row condition data
    condition_data = []
    for _, row in df_filtered.iterrows():
        file_id = row["id"]

        ts_str   = str(row.get("time_signatures", "4/4"))
        first_ts = ts_str.split(",")[0].strip() if ts_str != "nan" else "4/4"
        if "/" not in first_ts:
            first_ts = "4/4"

        inst         = "piano"
        num_measures = int(row.get("total_bars", 0))
        split        = str(row.get("split", "train"))
        if split not in ("train", "val", "test"):
            split = "train"

        condition_data.append({
            # ── Columns read by the dataloader (NECESSARY) ────────────────
            "file_base_name":           f"{file_id}.npy",
            "condition_file_base_name": f"{file_id}_condition.npy",
            "split":                    split,

            # ── Human-readable columns ─────────────────────────────────────
            "id":            file_id,
            "num_measures":  num_measures,
            "time_signature": first_ts,
            "inst":          inst,
        })

    df = pd.DataFrame(condition_data)

    # 4. Build indexed_tokens_dict ─────────────────────────────────────────
    indexed_tokens_dict = {
        "soc_token_compound": -4,
        "eoc_token_compound": -5,
    }

    # 4a. Metadata tokens (-6 downward)
    next_id = METADATA_TOKEN_START
    for col in METADATA_COLUMNS:
        for val in sorted(df[col].unique()):
            key = f"{col}_{val}"
            indexed_tokens_dict[key] = next_id
            next_id -= 1

    used_metadata_slots = abs(next_id - METADATA_TOKEN_START)
    if used_metadata_slots > NUM_METADATA_VALUES_MAX:
        raise ValueError(
            f"Too many metadata values: {used_metadata_slots} > "
            f"NUM_METADATA_VALUES_MAX={NUM_METADATA_VALUES_MAX}. "
            f"Increase num_metadata_values in model config or reduce METADATA_COLUMNS."
        )
    print(f"Metadata tokens: {used_metadata_slots} / {NUM_METADATA_VALUES_MAX} slots used")

    # 4b. Function tokens (-26 to -32, factored)
    for func_lbl in _FUNCTIONS:
        func_idx = FUNC_VOCAB[func_lbl]
        token_id = FUNC_TOKEN_START - func_idx          # -26, -27, ..., -32
        indexed_tokens_dict[f"func_{func_lbl}"] = token_id

    print(f"Function tokens: {len(_FUNCTIONS)} "
          f"(IDs {FUNC_TOKEN_START} to {FUNC_TOKEN_START - len(_FUNCTIONS) + 1})")

    # 4c. Section tokens (-33 to -59, factored, A-Z + 's')
    for sec_lbl in _SECTIONS:
        sec_idx  = SEC_VOCAB[sec_lbl]
        token_id = SEC_TOKEN_START - sec_idx            # -25, -26, ..., -51
        indexed_tokens_dict[f"sec_{sec_lbl}"] = token_id

    total_sec = len(_SECTIONS)
    print(f"Section  tokens: {total_sec} "
          f"(IDs {SEC_TOKEN_START} to {SEC_TOKEN_START - total_sec + 1})")
    print(f"Total structure tokens: {len(_FUNCTIONS) + total_sec}")

    # 5. Assign `label` column: list of metadata token IDs for each row
    def make_label(row):
        return str([indexed_tokens_dict[f"{col}_{row[col]}"] for col in METADATA_COLUMNS])

    df["label"] = df.apply(make_label, axis=1)

    # 6. Save indexed_tokens_dict.json
    TOKEN_DICT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_DICT_OUT, "w") as f:
        json.dump(indexed_tokens_dict, f, indent=2)
    print(f"\nSaved token dict ({len(indexed_tokens_dict)} entries) → {TOKEN_DICT_OUT}")

    # 7. Save condition.csv
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved condition CSV → {OUTPUT_CSV}")
    print(f"\nLabel preview (first 3 rows):")
    print(df[["id", "time_signature", "inst", "label"]].head(3).to_string(index=False))
    print(f"\nToken dict preview:")
    for k, v in indexed_tokens_dict.items():
        print(f"  {v:5d}  {k}")


if __name__ == "__main__":
    main()
