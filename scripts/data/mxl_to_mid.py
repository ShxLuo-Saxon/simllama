"""
Convert MXL files to MIDI using MuseScore 4 CLI.
Reads file list from metadata.xlsx (stratified subset) and batch converts.
"""

import argparse
import subprocess
import os
import pathlib
import pandas as pd
from tqdm import tqdm
from music21 import converter, bar, repeat, spanner, tempo as m21tempo

# -------------------------------------------------------------------------------------
# Global Variables
# -------------------------------------------------------------------------------------
MUSESCORE_EXE = pathlib.Path(os.environ.get("MUSESCORE", r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe"))
BASE_DIR = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[2]
                                 / "data" / "ninsheetmusic"))  # scores, metadata, processed/
METADATA_FILE = BASE_DIR / "metadata.xlsx"
MXL_DIR = BASE_DIR / "processed" / "mxl"
FIX_DIR = MXL_DIR / "fixes"
OUTPUT_DIR = BASE_DIR / "processed" / "mid"

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def get_stratified_ids(metadata_file: pathlib.Path) -> list[int]:
    """Read metadata and return IDs of files in the stratified subset."""
    df = pd.read_excel(metadata_file)
    is_stratified = df['subset'].str.contains('stratified', na=False)
    return df[is_stratified]['id'].tolist()

def find_mxl_file(mxl_dir: pathlib.Path, file_id: int) -> pathlib.Path | None:
    """Find the MXL file for a given ID."""
    mxl_path = mxl_dir / f"{file_id:05d}.mxl"
    return mxl_path if mxl_path.exists() else None

# -------------------------------------------------------------------------------------
# Core Function
# -------------------------------------------------------------------------------------
def convert_mxl_to_mid_musescore(musescore_exe: pathlib.Path, mxl_path: pathlib.Path, mid_path: pathlib.Path) -> bool:
    """Convert a single MXL file to MIDI using MuseScore CLI (Original Logic)."""
    try:
        result = subprocess.run(
            [str(musescore_exe), "-o", str(mid_path), str(mxl_path)],
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout per file
        )
        if result.returncode != 0:
            print(f"  MuseScore error: {result.stderr.strip()}")
            return False
        return mid_path.exists()
    except subprocess.TimeoutExpired:
        print(f"  Timeout converting {mxl_path.name}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False
        
def convert_mxl_to_mid_music21(mxl_path: pathlib.Path, mid_path: pathlib.Path) -> bool:
    """Convert a single MXL file to MIDI using music21, stripping all repeats and tempo changes.

    Tempo marks are stripped so music21 exports at a flat 120 BPM.  This makes
    calibrate_to_default_tempo=True in MusicTokenizer a no-op and keeps note onset
    ticks proportional to quarter-length offsets -- matching bar_onsets computed by
    structure_data_preprocess.get_bar_offsets() from the same MXL.  Without this,
    multi-tempo pieces accumulate onset drift after each tempo change event.
    """
    try:
        score = converter.parse(str(mxl_path))

        for p in score.parts:
            # 1. Standard Repeat Barlines and Text Expressions (D.C. al Coda, Segno)
            for el in p.recurse().getElementsByClass([bar.Repeat, repeat.RepeatExpression]):
                p.remove(el, recurse=True)
            # 2. 1st and 2nd Ending brackets
            for sp in p.recurse().getElementsByClass(spanner.RepeatBracket):
                p.remove(sp, recurse=True)
            # 3. Tempo marks -- strip so MIDI is exported at flat 120 BPM.
            #    calibrate_to_default_tempo=True then scales correctly for all bars.
            for el in p.recurse().getElementsByClass(m21tempo.MetronomeMark):
                p.remove(el, recurse=True)

        score.write('midi', fp=str(mid_path))
        return mid_path.exists()
    except Exception as e:
        print(f"  Error converting {mxl_path.name}: {e}")
        return False

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert MXL files to MIDI.")
    parser.add_argument('--ignore-repeats', action='store_true', help="Use music21 instead of MuseScore to ignore repeats and prevent MIDI loop unrolling.")
    args = parser.parse_args()

    # Verify MuseScore exists if not ignoring repeats
    if not args.ignore_repeats and not MUSESCORE_EXE.exists():
        print(f"Error: MuseScore not found at {MUSESCORE_EXE}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Get file IDs — all annotated files (authoritative list)
    ANNOTATION_DIR = BASE_DIR / "processed" / "annotation" / "annotations"
    stratified_ids = sorted([int(p.stem) for p in ANNOTATION_DIR.glob("*.json")])
    print(f"Found {len(stratified_ids)} annotated files to convert.")

    # Convert each file
    success_count = 0
    fail_count = 0
    skip_count = 0

    for file_id in tqdm(stratified_ids, desc="Converting MXL → MIDI"):
        mid_path = OUTPUT_DIR / f"{file_id:05d}.mid"

        # Find the MXL source file, check fix version first
        mxl_path = find_mxl_file(FIX_DIR, file_id)
        has_fix = mxl_path is not None
        if mxl_path is None:
            mxl_path = find_mxl_file(MXL_DIR, file_id)

        # Skip if already converted — unless a fixed MXL exists (always reconvert fixes)
        if mid_path.exists() and not has_fix:
            skip_count += 1
            continue
        if mxl_path is None:
            print(f"  Warning: No MXL file found for ID {file_id:05d}")
            fail_count += 1
            continue

        # Convert
        if args.ignore_repeats:
            success = convert_mxl_to_mid_music21(mxl_path, mid_path)
        else:
            success = convert_mxl_to_mid_musescore(MUSESCORE_EXE, mxl_path, mid_path)
            
        if success:
            success_count += 1
        else:
            print(f"  Failed: {mxl_path.name}")
            fail_count += 1

    print(f"\nDone! Success: {success_count}, Failed: {fail_count}, Skipped: {skip_count}")

if __name__ == "__main__":
    main()
