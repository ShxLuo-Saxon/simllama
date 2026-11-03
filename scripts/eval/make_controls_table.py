r"""
scripts/eval/make_controls_table.py

Rows of Table 3 (outputs/eval/controls.txt) and Table S6 (controls_qf.txt) in the paper's
layout: measures as rows, comparisons as columns, "share% (Holm p)", bold where Holm p < 0.05.

Usage (from the project root):
    .venv\Scripts\python.exe scripts\eval\make_controls_table.py [outputs/eval/controls_qf.txt]
"""
import re
import sys

BLOCKS = {"M1 high above M4": "High > No Label", "M4 above M1 low": "No Label > Low",
          "M4 above M1 mid (two-sided)": "No Label > Mid", "M3 high above low (two-sided)": "Wrong Label: High > Low",
          "M4 above M2": "No Label > Unconditioned"}
LINE = re.compile(r"\s+(\S+(?: \(ours\))?)\s+share ([\d.]+).*Holm ([\d.]+)")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/eval/controls.txt"
    res, counts = {}, {}
    for block in open(path, encoding="utf-8").read().split("\n\n"):
        lines = block.strip().split("\n")
        head = lines[0].split(" |")[0]
        if head not in BLOCKS:
            continue
        counts[BLOCKS[head]] = re.search(r"\| (\d+) contexts", lines[0]).group(1)
        for line in lines[1:]:
            m = LINE.match(line)
            res.setdefault(m.group(1), {})[BLOCKS[head]] = (float(m.group(2)), float(m.group(3)))
    cols = list(BLOCKS.values())
    print("| Measure | " + " | ".join(cols) + " |")
    print("|----------|-------------:|-------------:|-------------:|-------------:|---------------:|")
    for meas, v in res.items():
        cells = []
        for c in cols:
            share, holm = v[c]
            cell = f"{share * 100:.0f}% ({holm:.3f})"
            cells.append(f"**{cell}**" if holm < 0.05 else cell)
        print(f"| {meas} | " + " | ".join(cells) + " |")
    print("\ncontexts per column:", counts)


if __name__ == "__main__":
    main()
