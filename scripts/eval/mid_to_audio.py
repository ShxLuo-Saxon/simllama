"""
Summary: Render MIDI to audio (mp3/wav/ogg/flac) via MuseScore 4 CLI job file (-j)
Created: 2026-08-21
Last Updated: 2026-08-21
Version: 1.0
Status: stable
Scope: module

Render MIDI files to audio using MuseScore 4. MuseScore 4 requires a JSON job
file for CLI conversion (-j flag); this script builds one and invokes MuseScore
once per batch rather than once per file.

Output formats
--------------
mp3 (default), wav, ogg, flac. MuseScore infers the encoder from the output
file extension, so no codec flags are needed.

    mp3   web delivery, ~15 MB for 90 x 16 s clips  (default)
    ogg   web delivery, better quality per byte, less universal
    wav   lossless, ~10x larger -- local analysis, not for upload
    flac  lossless compressed

Usage:
    py mid_to_audio.py <file.mid> [out_dir] [--format ogg]
    py mid_to_audio.py --batch-dir <mid_dir> <out_dir> [--format wav]

Notes:
    MuseScore renders through its own synthesiser and soundfont, so output is
    reproducible on one machine but may differ across MuseScore versions or
    soundfont configurations. Record the version when the output is evidence.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MSCORE = Path(r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe")

FORMATS = ("mp3", "wav", "ogg", "flac")
DEFAULT_FORMAT = "mp3"


def _resolve_mscore(explicit: str | None = None) -> Path:
    """Locate the MuseScore binary, preferring an explicit path, then PATH."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"ERROR: MuseScore not found at {p}")
        return p
    if MSCORE.exists():
        return MSCORE
    for name in ("MuseScore4", "mscore4", "mscore", "musescore"):
        found = shutil.which(name)
        if found:
            return Path(found)
    sys.exit(
        f"ERROR: MuseScore not found at {MSCORE} or on PATH.\n"
        "       Pass --mscore <path/to/MuseScore4.exe>"
    )


def _run_job(pairs: list[tuple[Path, Path]], mscore: Path) -> list[str]:
    """
    Run MuseScore on a list of (in, out) path pairs via a job file.
    Returns the stems that produced no output file.
    """
    job = [{"in": str(src), "out": str(dst)} for src, dst in pairs]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(job, f)
        job_path = f.name

    try:
        result = subprocess.run([str(mscore), "-j", job_path],
                                capture_output=True, text=True)
    finally:
        Path(job_path).unlink(missing_ok=True)

    # A file that exists but is empty is a failed render, not a success.
    errors = [src.stem for src, dst in pairs
              if not dst.exists() or dst.stat().st_size == 0]
    if errors and result.stderr.strip():
        print(result.stderr.strip()[:400], file=sys.stderr)
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render MIDI to audio via MuseScore 4",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="A .mid file, or the input dir with --batch-dir")
    ap.add_argument("out_dir", nargs="?", help="Output dir (default: alongside input)")
    ap.add_argument("--batch-dir", action="store_true",
                    help="Treat `source` as a directory of .mid files")
    ap.add_argument("--format", default=DEFAULT_FORMAT, choices=FORMATS,
                    help=f"Output audio format (default: {DEFAULT_FORMAT})")
    ap.add_argument("--pattern", default="*.mid",
                    help="Glob for --batch-dir (default: *.mid)")
    ap.add_argument("--mscore", help="Path to the MuseScore binary")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render files whose output already exists")
    args = ap.parse_args()

    mscore = _resolve_mscore(args.mscore)
    src = Path(args.source)
    ext = args.format

    if args.batch_dir:
        if not src.is_dir():
            sys.exit(f"ERROR: not a directory: {src}")
        out_dir = Path(args.out_dir) if args.out_dir else src
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(src.glob(args.pattern))
        if not files:
            sys.exit(f"ERROR: no files matching {args.pattern} in {src}")

        pairs = [(f, out_dir / f"{f.stem}.{ext}") for f in files]
        if not args.overwrite:
            skipped = sum(1 for _, d in pairs if d.exists())
            pairs = [(s, d) for s, d in pairs if not d.exists()]
            if skipped:
                print(f"Skipping {skipped} already rendered (--overwrite to redo)")
        if not pairs:
            print("Nothing to do.")
            return

        print(f"Rendering {len(pairs)} files to {ext}...")
        errors = _run_job(pairs, mscore)

        print(f"Done: {len(pairs) - len(errors)} ok, {len(errors)} failed")
        if errors:
            print("Failed:", errors[:20], "..." if len(errors) > 20 else "")
            sys.exit(1)

    else:
        if not src.exists():
            sys.exit(f"ERROR: not found: {src}")
        out_dir = Path(args.out_dir) if args.out_dir else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src.stem}.{ext}"

        errors = _run_job([(src, out_path)], mscore)
        if errors:
            sys.exit(f"ERROR: render failed for {src.name}")
        size_kb = out_path.stat().st_size / 1024
        print(f"Written: {out_path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
