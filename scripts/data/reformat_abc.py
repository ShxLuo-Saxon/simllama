import sys
import re
import os
import argparse
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2]
                                 / "data" / "ninsheetmusic"))  # scores, metadata, processed/

# -------------------------------------------------------------------------------------
# Global Variables
# -------------------------------------------------------------------------------------
DEFAULT_INPUT_DIR = DATA_ROOT / "processed" / "abc"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "processed" / "rabc"

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def parse_abc(content):
    lines = content.splitlines()
    header_lines = []
    voice_content = {}
    current_voice = "1" # Default to voice 1
    
    # Simple heuristic to separate headers from body
    # Standard ABC headers start with a letter and a colon, e.g. X:1, T:Title
    # Voice definitions V:1 also look like headers but can appear in body.
    # Music lines do not start with Header: (usually).
    
    # We will treat everything as header until we hit a line that is NOT a field,
    # OR we hit a line that strictly looks like music note data.
    # However, V:1 lines can be interlaced in the body.
    
    # Better approach:
    # 1. Identify all Voice IDs used.
    # 2. Collect all "Header" lines (Global metadata).
    # 3. Collect "Body" lines for each voice.
    
    # We will walk through the lines.
    # State 0: Header Section.
    # State 1: Music Section.
    
    state = 0
    music_started = False
    
    # Regex for a field line: start of line, Letter, Colon.
    # Note: "w:" (lyrics) is attached to music.
    # "V:" is structural.
    field_pattern = re.compile(r'^[A-Za-z]:\s*.*')
    
    # We need to preserve the initial Voice definitions (e.g. V:1 treble...) in the header
    # but also switch voices in the body.
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if state == 0:
                header_lines.append(line)
            continue
            
        # Check if line is a comment (if starts with %, it's a comment line)
        if line.startswith('%'):
            if state == 0:
                header_lines.append(line)
            continue
            
        is_field = field_pattern.match(line)
        
        # Heuristic for start of music:
        # If it's NOT a field, it's music.
        # If it IS a field, but it is V:..., it changes voice.
        # But V: in header is just definition.
        # If we see a line that is definitely music (notes), we lock into Body mode.
        
        if not is_field:
            if not music_started:
                music_started = True 
                
        if not music_started:
            # Check if this is a V: line
            if line.startswith('V:'):
                parts = line.split()
                if len(parts) > 0:
                    v_val = line[2:].strip().split()[0]
                    current_voice = v_val
                    
                # If this V: line has no properties (just V:1), it might be redundant if it's the last thing.
                # But we don't know if it's the last thing yet.
                # We will store it.
                header_lines.append(line)
            else:
                header_lines.append(line)
        else:
            # We are in music body.
            if line.startswith('V:'):
                v_val = line[2:].strip().split()[0]
                current_voice = v_val
            else:
                # Append to current voice music buffer
                if current_voice not in voice_content:
                    voice_content[current_voice] = []
                # Remove inline linebreaks $ if present at end or inside
                clean_line = line.replace('$', '')
                voice_content[current_voice].append(clean_line)

    # Post-process headers to remove trailing simple V: lines that are just switches
    # A simple switch is "V: ID" or "V:ID" with no other text.
    
    # First, strip trailing empty lines from headers to find the real last line
    while header_lines and not header_lines[-1].strip():
        header_lines.pop()

    if header_lines:
        last_line = header_lines[-1]
        if last_line.startswith('V:'):
            # Check if it has properties (nm=, clef=, etc) or is just a number
            # Heuristic: split by space. If len=1 (V:1), or just number.
            # However some V lines are V:1 clef=treble.
            # If it is just "V:1", it's redundant because body starts with V:1.
            # Remove whitespace and V:
            content = last_line[2:].strip()
            # If content is just alphanumeric digits/chars with no spaces/equals?
            if ' ' not in content and '=' not in content:
                 header_lines.pop()
                 
    # Clean up trailing empty lines again just in case
    while header_lines and not header_lines[-1].strip():
        header_lines.pop()

    return header_lines, voice_content

def remove_comments(text):
    # Remove % comments, but respect quoted strings
    # We define a simple state machine
    out = []
    in_quote = False
    i = 0
    while i < len(text):
        c = text[i]
        if c == '"':
            in_quote = not in_quote
            out.append(c)
        elif c == '%' and not in_quote:
            # Comment starts, ignore rest of line
            break
        else:
            out.append(c)
        i += 1
    return "".join(out).strip()

def split_into_bars(text):
    # Split by bar delimiters.
    # Delimiters: | || |] |: :| ::
    # We want to keep the delimiters attached to the bar (at the end).
    
    # Regex to split but keep delimiter.
    # We ignore bars inside quoted strings (e.g. comments/annotations like "^|").
    # This is complex. Simplified approach:
    # 1. Hide quoted strings.
    # 2. Split.
    # 3. Restore.
    
    # Regex for bar lines:
    # We want to match [| or |] or || or |: or :| or :: or just |
    # Longest match first.
    # Pattern: (::|:\||\|:|\|\||\|]|\[\||\|)
    
    # But wait, we need to handle "text with | inside".
    # Let's iterate through the string.
    
    bars = []
    current_bar = []
    in_quote = False
    i = 0
    n = len(text)
    
    while i < n:
        char = text[i]
        
        if char == '"':
            in_quote = not in_quote
            current_bar.append(char)
            i += 1
            continue
            
        if not in_quote:
            # Check for bar patterns starting at i
            # Lookahead for max length 2 (::, ||, |], [|, |:, :|)
            
            matched = False
            for length in [2, 1]:
                if i + length <= n:
                    sub = text[i:i+length]
                    is_delim = False
                    if length == 2:
                        if sub in ['::', ':|', '|:', '||', '|]', '[|']:
                            is_delim = True
                    elif length == 1:
                        if sub == '|':
                            is_delim = True
                    
                    if is_delim:
                        # Found a delimiter.
                        # If current_bar is empty, this is a barline at the start,
                        # so include it as part of the first bar instead of creating an empty bar
                        if current_bar:
                            # Normal case: append delimiter and flush
                            current_bar.append(sub)
                            bars.append("".join(current_bar).strip())
                            current_bar = []
                        else:
                            # Starting barline: include it in the next bar
                            current_bar.append(sub)
                        
                        i += length
                        matched = True
                        break
            
            if matched:
                continue
        
        current_bar.append(char)
        i += 1
        
    if current_bar:
        rem = "".join(current_bar).strip()
        if rem:
            bars.append(rem)
    
    # Filter out bars that contain only delimiters and no musical content
    # A bar is empty if it only contains bar line symbols: | || |: :| :: |] [|
    def has_music_content(bar):
        # Remove all bar delimiters and whitespace
        cleaned = bar.replace('|:', '').replace(':|', '').replace('::', '')
        cleaned = cleaned.replace('||', '').replace('|]', '').replace('[|', '')
        cleaned = cleaned.replace('|', '').strip()
        return len(cleaned) > 0
    
    # Keep bars that have music content (including those that start with barlines)
    bars = [b for b in bars if has_music_content(b)]
    
    # Remove completely empty bars (only whitespace)
    bars = [b.strip() for b in bars if b.strip()]
            
    return bars

def extract_original_bar_count(content):
    """Extract bar count from original ABC file by finding highest inline bar number like %216"""
    
    # Find all inline bar number comments (e.g., "| d3 :| %216")
    # Pattern: % followed by digits, possibly at end of line
    inline_pattern = re.compile(r'%(\d+)\s*$', re.MULTILINE)
    bar_numbers = [int(m.group(1)) for m in inline_pattern.finditer(content)]
    
    if bar_numbers:
        return max(bar_numbers)
    
    return None



# -------------------------------------------------------------------------------------
# Core Function
# -------------------------------------------------------------------------------------
def process_file(filepath, output_dir):
    filepath = Path(filepath)
    output_dir = Path(output_dir)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract original bar count if available
    original_bar_count = extract_original_bar_count(content)
        
    headers, voice_content = parse_abc(content)
    
    # Keep the original T: field from the ABC file
    # (Title is no longer synced with filename)
    
    # Process voices
    # Join all lines for each voice into one string
    voice_bars = {}
    max_bars = 0
    
    sorted_voices = sorted(voice_content.keys())
    
    for v in sorted_voices:
        # Pre-process lines to remove comments
        if v in voice_content:
            clean_lines = [
                remove_comments(l)
                for l in voice_content[v]
                if not l.lstrip().startswith(("w:", "W:"))
            ]
            full_text = " ".join(clean_lines) 
            
            bars = split_into_bars(full_text)
            
            # --- UPDATE: Sanitize 2nd and 3rd ending bars ---
            # If a bar starts with '2' or '3' (second/third ending) and ends with '||', convert to '|'
            sanitized_bars = []
            for b in bars:
                # Regex: Start of line, optional [, 2 or 3, not a digit
                if re.match(r'^\s*\[?[23](?![0-9])', b) and b.endswith('||'):
                    b = b[:-2] + '|'
                sanitized_bars.append(b)
            voice_bars[v] = sanitized_bars
            
            if len(sanitized_bars) > max_bars:
                max_bars = len(sanitized_bars)
            

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = filepath.name
    base_name, ext = filepath.stem, filepath.suffix
    out_path = output_dir / f"{base_name}{ext}"
    
    # Validate bar count
    bar_match = None
    if original_bar_count is not None:
        bar_match = (max_bars == original_bar_count)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        # Write headers
        for h in headers:
            f.write(h + "\n")
        # Add bar count metadata
        f.write(f"% numOfBars: {max_bars}\n")
        f.write("\n")
        
        # Write bars
        for i in range(max_bars):
            # Check if this bar index actually has content in any voice?
            f.write(f"% Bar {i+1}\n")
            for v in sorted_voices:
                bars = voice_bars.get(v, [])
                if i < len(bars):
                    bar_content = bars[i]
                    f.write(f"V:{v}\n")
                    f.write(bar_content + "\n")
            f.write("\n")
    
    # Return statistics for summary
    return {
        'filename': filepath.name,
        'bars': max_bars,
        'original_bars': original_bar_count,
        'bar_match': bar_match,
        'voices': len(sorted_voices)
    }

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reformat ABC files into bar-aligned rabc files."
    )
    parser.add_argument(
        "input", 
        nargs='?', 
        default=DEFAULT_INPUT_DIR, 
        help="Input .abc file or directory"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        default=DEFAULT_OUTPUT_DIR, 
        help="Output directory for reformatted .abc"
    )
    args = parser.parse_args()

    target = Path(args.input)
    output_dir = Path(args.output)
    
    stats = []
    if target.is_file():
        result = process_file(target, output_dir)
        if result:
            stats.append(result)
    elif target.is_dir():
        for file_path in target.rglob("*"):
            if file_path.suffix.lower() == ".abc":
                result = process_file(file_path, output_dir)
                if result:
                    stats.append(result)
    
    # Summary
    if stats:
        df = pd.DataFrame(stats)
        summary_path = output_dir / "summary.csv"
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        print(f"Total files processed: {len(stats)}")
        print(f"Average bars: {df['bars'].mean():.2f}")
        print(f"Bar match rate: {df['bar_match'].mean() * 100:.2f}%")
        print(f"Average voices: {df['voices'].mean():.2f}")


if __name__ == "__main__":
    main()