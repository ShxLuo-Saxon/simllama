r"""Derive and apply score-free bar maps for the SimLlama release.

Matches archived hand-unrolled measures against original scores using every
part's note/rest offsets, pitches and durations. Reads compressed scores in
memory; writes only the map CSV during derive. Tests written and played sources.

Usage (from the project root):
    py scripts/release/bar_maps.py derive
    py scripts/release/bar_maps.py apply --id 00001 --raw downloaded.mxl --out unrolled.mxl
    py scripts/release/bar_maps.py derive --literal-ids --pairs pairs.csv --sections sections --raw-dir raw --fix-zip backup.zip
    # Private annotation archive, supplied explicitly:
    py scripts/release/bar_maps.py derive --raw-dir "<scores>/mxl_raw" --fix-zip "<scores>/_backup/2026-09-29_pre-mxl-split.zip" --metadata "<annotation project>/dataset/metadata.csv"

Bar orders use inclusive ranges, for example "1-4,5-12,5-12,13-20".
A pickup is bar 0. Counts exclude it for compatibility with round 1.
The preprocessing code counts physical measures from 1, including pickups.
Derivation also accepts a complete physical-measure count, recording this choice.
Section endpoints can omit trailing X/s sections; count equality alone cannot
prove that the historical MIDI or section tokens followed this play order.
Identical bars can have several valid origins; matching favours consecutive runs.
Verification concerns musical content, not engraving, ties, dynamics or navigation.
For played sources, verification compares every part against the expanded score.
music21 supplies the primary expansion. A bounded interpreter supplies a fallback
for repeat barlines, voltas, and D.S./D.C. al Fine/Coda, without repeats after jumps.
Fallback results identify their method in the note; they do not validate MuseScore.
Orders always reference the original raw score, including for fix_played.
Training IDs differ from sheet IDs. The default metadata crosswalk joins exact
original_filename values; the resolved sheet ID is recorded in each row's note.
Use --literal-ids only when score filenames already use the training IDs.

Version: 2.0
Last Updated: 2026-10-31
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from music21 import bar, clef, converter, expressions, key, meter, repeat, spanner, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP = ROOT / "outputs/release/bar_maps.csv"
FIELDS = "piece_id source fix_date bars_raw bars_used order verified note".split()
SECTION = re.compile(r"^(.*)_s\d+_.+_(\d+)-(\d+)\.npy$")


def read_score(data: Path | bytes) -> stream.Score:
    """Parse XML or compressed MusicXML without disk extraction or parser caches."""
    payload = data.read_bytes() if isinstance(data, Path) else data
    if zipfile.is_zipfile(io.BytesIO(payload)):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            container = ET.fromstring(archive.read("META-INF/container.xml"))
            names = [n.attrib["full-path"] for n in container.iter()
                     if n.tag.split("}")[-1] == "rootfile"]
            payload = archive.read(names[0])
    return converter.parseData(payload, format="musicxml")


def measures(score):
    parts = [list(p.getElementsByClass(stream.Measure)) for p in score.parts]
    if not parts or not parts[0]:
        raise ValueError("score has no part measures")
    if any(len(p) != len(parts[0]) for p in parts):
        raise ValueError("parts have different measure counts")
    return parts


def first_bar(score):
    """Use music21's explicit pickup numbering; otherwise number from one."""
    return 0 if measures(score)[0][0].number == 0 else 1


def bar_count(score):
    return len(measures(score)[0]) - (first_bar(score) == 0)


def signatures(score):
    """Return per-bar tuples of per-part note/rest multisets, including grace notes."""
    parts = measures(score)
    result = []
    for index in range(len(parts[0])):
        voices = []
        for part in parts:
            measure = part[index]
            events = []
            for event in measure.recurse().notesAndRests:
                offset = Fraction(event.getOffsetInHierarchy(measure)).limit_denominator(1000000)
                duration = Fraction(event.duration.quarterLength).limit_denominator(1000000)
                pitches = ("rest",) if event.isRest else tuple(p.nameWithOctave for p in event.pitches)
                for pitch in pitches:
                    events.append((offset, pitch, duration))
            voices.append(tuple(sorted(events)))
        result.append(tuple(voices))
    return result


def parse_order(text):
    result = []
    for token in text.split(","):
        match = re.fullmatch(r"\s*(\d+)(?:\s*-\s*(\d+))?\s*", token)
        if not match:
            raise ValueError(f"invalid order token: {token!r}")
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start:
            raise ValueError(f"descending order range: {token!r}")
        result.extend(range(start, end + 1))
    return result


def compress_order(order):
    # End each written-out pass explicitly, including its final occurrence.
    boundaries = {a for a, b in zip(order, order[1:]) if b <= a}
    groups = []
    start = previous = order[0]
    for value in order[1:]:
        if value != previous + 1 or previous in boundaries:
            groups.append(str(start) if start == previous else f"{start}-{previous}")
            start = value
        previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def match_order(raw, fixed):
    """Find an exact map, minimizing jumps across all ambiguous matching bars."""
    original, target = signatures(raw), signatures(fixed)
    candidates = defaultdict(list)
    for index, signature in enumerate(original):
        candidates[signature].append(index + first_bar(raw))
    states = {None: (0, ())}
    ambiguous = 0
    for index, signature in enumerate(target):
        options = candidates[signature]
        if not options:
            return None, f"first differing bar {index + first_bar(fixed)}: no identical raw bar"
        ambiguous += len(options) > 1
        next_states = {}
        for value in options:
            next_states[value] = min(
                (cost + (previous is not None and value != previous + 1), path + (value,))
                for previous, (cost, path) in states.items())
        states = next_states
    order = list(min(states.values())[1])
    return order, (f"{ambiguous} bars have multiple identical raw candidates; continuity breaks ties"
                   if ambiguous else "")


def build_score(raw, order):
    """Copy measures in order and remove playback navigation from the result."""
    parts = measures(raw)
    start = first_bar(raw)
    if not order or any(n < start or n >= start + len(parts[0]) for n in order):
        raise ValueError("order references a missing raw bar")
    result = stream.Score()
    result.metadata = copy.deepcopy(raw.metadata)
    for source, bars in zip(raw.parts, parts):
        dest = stream.Part(id=str(source.id))
        dest.partName = source.partName
        dest.partAbbreviation = source.partAbbreviation
        for item in source:
            if not isinstance(item, (stream.Measure, spanner.Spanner)):
                dest.insert(item.offset, copy.deepcopy(item))
        for index, number in enumerate(order):
            original = bars[number - start]
            measure = copy.deepcopy(original)
            measure.number = index + (0 if order[0] == 0 else 1)
            measure.numberSuffix = None
            # Restore inherited context when a jump skips the original declaration.
            for cls in (clef.Clef, key.KeySignature, meter.TimeSignature):
                if not measure.getElementsByClass(cls):
                    context = original.getContextByClass(cls)
                    if context is not None:
                        measure.insert(0, copy.deepcopy(context))
            for item in list(measure.recurse()):
                if isinstance(item, (repeat.RepeatExpression, spanner.Spanner)):
                    item.activeSite.remove(item)
            for side in ("leftBarline", "rightBarline"):
                if isinstance(getattr(measure, side), bar.Repeat):
                    setattr(measure, side, bar.Barline("regular"))
            dest.append(measure)
        result.insert(0, dest)
    return result


def verify(raw, fixed, order):
    actual, expected = signatures(build_score(raw, order)), signatures(fixed)
    for index in range(max(len(actual), len(expected))):
        if index >= min(len(actual), len(expected)) or actual[index] != expected[index]:
            return "no", f"first differing bar {index + first_bar(fixed)}: applied content differs"
    return "yes", ""


def navigation_order(score):
    """Interpret bar-level navigation shared across staves; reject ambiguous jumps."""
    parts = measures(score)
    size = len(parts[0])
    starts, ends, marks, endings = set(), {}, defaultdict(set), {}
    commands = {}
    for part, bars in zip(score.parts, parts):
        positions = {id(m): i for i, m in enumerate(bars)}
        for bracket in part.recurse().getElementsByClass(spanner.RepeatBracket):
            indices = [positions[id(m)] for m in bracket.getSpannedElements() if id(m) in positions]
            if not indices or not bracket.numberRange:
                raise ValueError("volta has no measure span or pass numbers")
            span = (min(indices), max(indices))
            passes = tuple(bracket.numberRange)
            if span in endings and endings[span] != passes:
                raise ValueError("conflicting volta numbers across staves")
            endings[span] = passes
        for i, measure in enumerate(bars):
            if isinstance(measure.leftBarline, bar.Repeat):
                if measure.leftBarline.direction != "start":
                    raise ValueError("unsupported left repeat direction")
                starts.add(i)
            if isinstance(measure.rightBarline, bar.Repeat):
                if measure.rightBarline.direction != "end":
                    raise ValueError("unsupported right repeat direction")
                times = measure.rightBarline.times or 2
                if i in ends and ends[i] != times:
                    raise ValueError("conflicting repeat counts across staves")
                ends[i] = times
            for item in measure.recurse():
                if isinstance(item, repeat.RepeatExpressionCommand):
                    name = type(item).__name__
                    if not name.startswith(("DaCapo", "DalSegno")):
                        raise ValueError(f"unsupported navigation command {name}")
                    command = (name, item.repeatAfterJump)
                    if i in commands and commands[i] != command:
                        raise ValueError("conflicting navigation commands across staves")
                    commands[i] = command
                elif isinstance(item, (repeat.Fine, repeat.Segno, repeat.Coda)):
                    marks[type(item).__name__].add(i)
                elif isinstance(item, expressions.TextExpression):
                    text = re.sub(r"[^a-z]", "", item.content.lower())
                    # music21 does not recognize every spelling of navigation words.
                    if text.startswith(("dc", "dacapo", "ds", "dalsegno")):
                        name = "DaCapo" if text.startswith(("dc", "dacapo")) else "DalSegno"
                        name += "AlFine" if "fine" in text else "AlCoda" if "coda" in text else ""
                        command = (name, False)
                        if i in commands and commands[i] != command:
                            raise ValueError("conflicting textual navigation commands")
                        commands[i] = command
                    elif text in ("fine", "segno", "coda", "tocoda", "alcoda"):
                        marks["Fine" if text == "fine" else "Segno" if text == "segno" else "Coda"].add(i)
    # Pair repeat ends with the nearest still-open start, including implicit starts.
    stack, spans, implicit = [], {}, 0
    for i in range(size):
        if i in starts:
            stack.append(i)
        if i in ends:
            start = stack.pop() if stack else implicit
            spans[i] = (start, ends[i])
            implicit = i + 1
    if stack:
        raise ValueError("repeat start has no closing barline")
    volta = {}
    previous_end, previous_span_end = None, -1
    for (start, end), passes in sorted(endings.items()):
        owners = [r for r in spans if start <= r <= end]
        owner = owners[0] if len(owners) == 1 else previous_end
        if owner is None or len(owners) > 1:
            raise ValueError("cannot associate volta with a repeat")
        if not owners and start != previous_span_end + 1:
            raise ValueError("nonadjacent volta has no closing repeat")
        volta[start] = (end, passes, owner)
        previous_end, previous_span_end = owner, end
        repeat_start, times = spans[owner]
        spans[owner] = (repeat_start, max(times, max(passes)))
    visits = defaultdict(lambda: 1)
    order, i, jumped, mode, repeats_after_jump = [], 0, False, "", False
    coda_taken = False
    for _ in range(max(1000, size * 100)):
        if i >= size:
            if jumped and mode and not (mode == "Coda" and coda_taken):
                raise ValueError(f"jump never reached its {mode} target")
            return [n + first_bar(score) for n in order]
        if i in volta:
            end, passes, owner = volta[i]
            current = spans[owner][1] if jumped and not repeats_after_jump else visits[owner]
            if current not in passes:
                i = end + 1
                continue
        order.append(i)
        if jumped and mode == "Fine" and i in marks["Fine"]:
            return [n + first_bar(score) for n in order]
        if jumped and mode == "Coda" and not coda_taken and i in marks["Coda"]:
            codas = sorted(marks["Coda"])
            if len(codas) != 2 or i != codas[0]:
                raise ValueError("al Coda needs exactly two ordered coda signs")
            i, coda_taken = codas[1], True
            continue
        if i in spans and (not jumped or repeats_after_jump):
            start, times = spans[i]
            if visits[i] < times:
                visits[i] += 1
                for inner, (inner_start, _) in spans.items():
                    if start <= inner_start and inner < i:
                        visits[inner] = 1
                i = start
                continue
        if i in commands and not jumped:
            name, repeats_after_jump = commands[i]
            mode = "Fine" if "Fine" in name else "Coda" if "Coda" in name else ""
            if name.startswith("DalSegno"):
                signs = sorted(marks["Segno"])
                if len(signs) != 1 or signs[0] > i:
                    raise ValueError("D.S. needs one preceding segno")
                i = signs[0]
            else:
                i = 0
            jumped = True
            if repeats_after_jump:
                visits.clear()
            continue
        i += 1
    raise ValueError("navigation exceeds bounded playback limit")


def played_candidates(score):
    """Return expanded scores with provenance orders and explicit method diagnostics."""
    candidates, diagnostics = [], []
    # Editorial provenance survives deepcopy and music21 expansion, even for identical bars.
    tagged = copy.deepcopy(score)
    for part in measures(tagged):
        for index, measure in enumerate(part):
            measure.editorial["bar_map_origin"] = index + first_bar(score)
    try:
        expanded = tagged.expandRepeats()
        parts = measures(expanded)
        order = [m.editorial["bar_map_origin"] for m in parts[0]]
        if any([m.editorial["bar_map_origin"] for m in p] != order for p in parts[1:]):
            raise ValueError("parts have different playback orders")
        status, problem = verify(score, expanded, order)
        if status != "yes":
            raise ValueError(problem)
        candidates.append((expanded, order, "music21"))
        diagnostics.append(f"music21={bar_count(expanded)}")
    except Exception as exc:
        diagnostics.append(f"music21 failed: {type(exc).__name__}: {exc}")
    try:
        order = navigation_order(score)
        if not any(order == previous for _, previous, _ in candidates):
            expanded = build_score(score, order)
            candidates.append((expanded, order, "navigation fallback"))
            diagnostics.append(f"navigation fallback={bar_count(expanded)}")
    except Exception as exc:
        diagnostics.append(f"navigation fallback failed: {type(exc).__name__}: {exc}")
    return candidates, "; ".join(diagnostics)


def write_score(score, output):
    """Export entirely in memory, avoiding music21's temporary score files."""
    exporter = GeneralObjectExporter(score)
    exporter.makeNotation = False  # Preserve existing rests and durations exactly.
    xml = exporter.parse()
    expected, actual = signatures(score), signatures(read_score(xml))
    for index in range(max(len(expected), len(actual))):
        if index >= min(len(expected), len(actual)) or expected[index] != actual[index]:
            raise ValueError(f"MusicXML export changes musical content at bar {index + first_bar(score)}")
    output = Path(output)
    if output.suffix.lower() == ".mxl":
        container = (b'<?xml version="1.0"?><container><rootfiles>'
                     b'<rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/>'
                     b'</rootfiles></container>')
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("META-INF/container.xml", container)
            archive.writestr("score.musicxml", xml)
    else:
        output.write_bytes(xml)


def section_ends(pairs, sections):
    """Resolve filename prefixes through both filename columns, then scan all sections."""
    prefixes, ends, referenced = {}, {}, set()
    with Path(pairs).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            piece = row["piece_id"]
            ends.setdefault(piece, 0)
            for column in ("context_file", "target_file"):
                name = row[column].replace(chr(92), "/").rsplit("/", 1)[-1]
                match = SECTION.fullmatch(name)
                if not match:
                    raise ValueError(f"unrecognized section filename: {name}")
                prefix = match[1]
                if prefix in prefixes and prefixes[prefix] != piece:
                    raise ValueError(f"section prefix {prefix} maps to multiple piece IDs")
                prefixes[prefix] = piece
                referenced.add(name)
    present = set()
    for path in Path(sections).glob("*.npy"):
        present.add(path.name)
        match = SECTION.fullmatch(path.name)
        if match and match[1] in prefixes:
            piece = prefixes[match[1]]
            ends[piece] = max(ends[piece], int(match[3]))
    missing = referenced - present
    if missing:
        raise ValueError(f"{len(missing)} referenced section files missing, including {min(missing)}")
    if any(end == 0 for end in ends.values()):
        raise ValueError("piece has no section endpoint")
    return ends


def source_crosswalk(piece_metadata, sheet_metadata):
    """Join training IDs to sheet IDs by exact titles; reject missing/ambiguous joins."""
    with zipfile.ZipFile(piece_metadata) as archive:
        workbook = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(n.itertext()) for n in strings]
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    records = []
    for row in workbook.findall("s:sheetData/s:row", ns):
        values = {}
        for cell in row.findall("s:c", ns):
            column = re.sub(r"\d+", "", cell.attrib["r"])
            value = cell.findtext("s:v", default="", namespaces=ns)
            if cell.attrib.get("t") == "inlineStr":
                value = "".join(n.text or "" for n in cell.findall(".//s:t", ns))
            elif cell.attrib.get("t") == "s":
                value = shared[int(value)]
            values[column] = value
        records.append(values)
    headers = {value: column for column, value in records[0].items()}
    titles = defaultdict(set)
    with Path(sheet_metadata).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            titles[row["original_filename"]].add(row["sheet_id"].strip())
    result = {}
    for row in records[1:]:
        piece = str(int(row[headers["id"]])).zfill(5)
        title = row.get(headers["original_filename"], "")
        options = titles[title]
        result[piece] = (next(iter(options)) if len(options) == 1 else None)
    return result


def annotation_ends(folder):
    """Last annotated bar per piece: the sections of a piece may stop before its end."""
    ends = {}
    for f in Path(folder).glob("*.json"):
        sections = json.loads(f.read_text(encoding="utf-8")).get("Section") or []
        if sections:
            ends[f.stem] = max(int(s["BarRange"][1]) for s in sections)
    return ends


def derive(pairs, sections, raw_dir, fix_zip, output, source_ids=None, annotations=None):
    ends = section_ends(pairs, sections)
    if annotations:                     # bars used = the later of the last section and the annotation's end
        ann = annotation_ends(annotations)
        ends = {p: max(e, ann.get(f"{int(p):05d}", 0)) for p, e in ends.items()}
    rows = []
    audit = defaultdict(Counter)
    with zipfile.ZipFile(fix_zip) as archive:
        for index, (piece, used) in enumerate(sorted(ends.items()), 1):
            row = dict.fromkeys(FIELDS, "")
            row.update(piece_id=piece, source="unknown", bars_used=used)
            sid = str(int(piece)) if source_ids is None else source_ids.get(piece)
            member = f"mxl/{sid}_fix.mxl"
            try:
                if sid is None:
                    raise ValueError("no unique original_filename match from training ID to sheet ID")
                raw = read_score(Path(raw_dir) / f"{sid}.mxl")
                row["bars_raw"] = bar_count(raw)
                fixed = read_score(archive.read(member)) if member in archive.namelist() else None
                fix_matches = fixed is not None and bar_count(fixed) == used
                raw_matches = bar_count(raw) == used
                if raw_matches and (not fix_matches or signatures(raw) == signatures(fixed)):
                    row["source"] = "raw"
                    row["verified"] = "yes"
                    if fixed is not None and not fix_matches:
                        row["note"] = f"archive fix has {bar_count(fixed)} bars; raw matches sections"
                else:
                    # Audit both possible historical sources, including unverified fixes.
                    raw_played, raw_diagnostic = played_candidates(raw)
                    fix_played, fix_diagnostic = played_candidates(fixed) if fixed is not None else ([], "absent")
                    diagnostic = f"played raw [{raw_diagnostic}]; played fix [{fix_diagnostic}]"
                    raw_options = [(s, o, method) for s, o, method in raw_played if bar_count(s) == used]
                    fix_options = [(s, o, method) for s, o, method in fix_played if bar_count(s) == used]
                    cohort = "unknown"
                    if fix_matches:
                        target, source, method = fixed, "fix", "written"
                    elif len(measures(raw)[0]) == used and (
                            fixed is None or len(measures(fixed)[0]) != used
                            or signatures(raw) == signatures(fixed)):
                        target, source, method = raw, "raw", "physical measures, pickup counts as bar 1"
                    elif fixed is not None and len(measures(fixed)[0]) == used:
                        target, source, method = fixed, "fix", "physical measures, pickup counts as bar 1"
                    elif fix_options and (not raw_options or signatures(fix_options[0][0]) != signatures(raw_options[0][0])):
                        target, _, method = fix_options[0]
                        source = "fix_played"
                    elif raw_options:
                        target, _, method = raw_options[0]
                        source = "raw_played"
                    elif fix_options:
                        target, _, method = fix_options[0]
                        source = "fix_played"
                    else:
                        target, source, method = None, "unknown", ""
                    if target is not None:
                        row["source"] = source
                        if source.startswith("fix"):
                            row["fix_date"] = "-".join(f"{n:02d}" for n in archive.getinfo(member).date_time[:3])
                        # Preserve provenance rather than inferring a path through identical bars.
                        if source == "raw":
                            order = list(range(first_bar(raw), first_bar(raw) + len(measures(raw)[0])))
                            note = ""
                        elif source == "raw_played":
                            order = raw_options[0][1]
                            note = ""
                        else:
                            order, note = match_order(raw, target)
                        row["verified"] = "no"
                        problem = ""
                        if order is not None:
                            serialized = compress_order(order)
                            row["order"] = serialized if source != "raw" else ""
                            row["verified"], problem = verify(raw, target, parse_order(serialized))
                        row["note"] = "; ".join(x for x in (problem, note, f"target={source} ({method})", diagnostic) if x)
                        if fix_matches:
                            cohort = "unverified_fix" if row["verified"] == "no" else "verified_fix"
                    else:
                        count = bar_count(fixed) if fixed is not None else "absent"
                        row["verified"] = "no"
                        row["note"] = f"bar-count mismatch: raw={bar_count(raw)}, fix={count}, used={used}; {diagnostic}"
                        physical = len(measures(raw)[0])
                        if first_bar(raw) == 0 and physical == used:
                            row["note"] += "; physical count matches when pickup counts as bar 1"
                        elif used < physical:
                            row["note"] += f"; endpoint precedes written end by {physical - used} physical measures (trailing X/s sections may be omitted)"
                        if fixed is not None and first_bar(fixed) == 0 and len(measures(fixed)[0]) == used:
                            row["note"] += "; fix physical count matches when pickup counts as bar 1"
                    primary = any(bar_count(s) == used and method == "music21"
                                  for s, _, method in raw_played + fix_played)
                    fallback = any(bar_count(s) == used and method != "music21"
                                   for s, _, method in raw_played + fix_played)
                    outcome = "music21 count match" if primary else "fallback count match" if fallback else "neither count matches"
                    audit[cohort][outcome] += 1
            except Exception as exc:
                row["note"] = f"{type(exc).__name__}: {exc}"
                row["verified"] = "no"
            if source_ids is not None and sid is not None:
                row["note"] = "; ".join(x for x in (f"sheet_id={sid} (exact original_filename match)", row["note"]) if x)
            rows.append(row)
            if index % 20 == 0:
                print(f"Processed {index}/{len(ends)} pieces", flush=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Pieces: {len(rows)}")
    for source in ("raw", "raw_played", "fix", "fix_played", "unknown"):
        verified = Counter(row["verified"] for row in rows if row["source"] == source)
        print(f"{source}: yes={verified['yes']}, no={verified['no']}")
    for cohort, outcomes in sorted(audit.items()):
        print(f"Played count audit ({cohort}): " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    fixes = [row for row in rows if row["source"].startswith("fix")]
    verified = Counter(row["verified"] for row in fixes)
    print(f"Fix verification: yes={verified['yes']}, no={verified['no']}")
    dates = Counter(row["fix_date"] for row in fixes)
    print("Used fix dates: " + ", ".join(f"{day}={count}" for day, count in sorted(dates.items())))
    print(f"Before 2026-04-15: {sum(n for day, n in dates.items() if day and day < '2026-04-15')}; "
          f"on: {dates['2026-04-15']}; after: {sum(n for day, n in dates.items() if day > '2026-04-15')}")
    print(f"Map: {output}")
    return rows


def apply_map(piece, raw_path, output, maps=DEFAULT_MAP):
    with Path(maps).open(encoding="utf-8-sig", newline="") as handle:
        matches = [r for r in csv.DictReader(handle) if int(r["piece_id"]) == int(piece)]
    if len(matches) != 1:
        raise ValueError(f"expected one map for piece {piece}; found {len(matches)}")
    row = matches[0]
    if row["source"] not in ("raw", "raw_played", "fix", "fix_played") or (
            row["source"] != "raw" and row["verified"] != "yes"):
        raise ValueError(f"piece {piece} has no verified map: {row['note']}")
    if Path(raw_path).resolve() == Path(output).resolve():
        raise ValueError("output would overwrite the raw score; choose another path")
    raw = read_score(Path(raw_path))
    if bar_count(raw) != int(row["bars_raw"]):
        raise ValueError("downloaded score bar count differs from the mapped original")
    order = (parse_order(row["order"]) if row["source"] != "raw"
             else list(range(first_bar(raw), len(measures(raw)[0]) + first_bar(raw))))
    write_score(build_score(raw, order), output)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    derive_parser = commands.add_parser("derive")
    derive_parser.add_argument("--pairs", type=Path, default=ROOT / "data/sections/pairs.csv")
    derive_parser.add_argument("--sections", type=Path, default=ROOT / "data/sections")
    derive_parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/scores/mxl_raw")
    derive_parser.add_argument("--fix-zip", type=Path, default=ROOT / "data/scores/_backup/2026-09-29_pre-mxl-split.zip")
    derive_parser.add_argument("--out", type=Path, default=DEFAULT_MAP)
    derive_parser.add_argument("--piece-metadata", type=Path, default=ROOT / "data/metadata.xlsx")
    derive_parser.add_argument("--metadata", "--sheet-metadata", dest="metadata", type=Path,
                               default=ROOT / "data/scores/metadata.csv",
                               help="sheet metadata CSV; joined to --piece-metadata by original_filename")
    derive_parser.add_argument("--literal-ids", action="store_true", help="score filenames use training piece IDs")
    derive_parser.add_argument("--annotations", type=Path, default=ROOT / "data/annotations/gemini_prompt_v1",
                               help="annotation JSON folder; its last bar counts too (sections may stop early)")
    apply_parser = commands.add_parser("apply")
    apply_parser.add_argument("--id", required=True)
    apply_parser.add_argument("--raw", type=Path, required=True)
    apply_parser.add_argument("--out", type=Path, required=True)
    apply_parser.add_argument("--maps", type=Path, default=DEFAULT_MAP)
    args = parser.parse_args()
    if args.command == "derive":
        source_ids = None if args.literal_ids else source_crosswalk(args.piece_metadata, args.metadata)
        derive(args.pairs, args.sections, args.raw_dir, args.fix_zip, args.out, source_ids, args.annotations)
    else:
        apply_map(args.id, args.raw, args.out, args.maps)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
