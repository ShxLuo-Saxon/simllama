# To run this code you need to install the following dependencies:
# pip install google-genai

import os
import json
import pathlib
from google import genai
from google.genai import types
from shared_config import get_response_schema, load_system_instruction, MODEL_ID



def generate(file_id, content_text, system_instruction_text):
    client = genai.Client(
        api_key=os.environ.get("GOOGLE_API_KEY_MSA"),
        http_options={
            "timeout": 180000,  # milliseconds, i.e., 3 minutes
        },
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=content_text),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        ),
        system_instruction=[
            types.Part.from_text(text=system_instruction_text),
        ],
        response_mime_type="application/json",
        response_schema=get_response_schema(),
    )

    # Collect the streamed response
    full_response = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_ID,
        contents=contents,
        config=generate_content_config,
    ):
        # print(chunk.text, end="", flush=True)
        full_response += chunk.text
    
    
    return full_response


if __name__ == "__main__":

    # Load system instruction
    system_instruction_text = load_system_instruction()

    # Paths
    BASE_DIR = pathlib.Path(os.environ.get("DATA_ROOT", pathlib.Path(__file__).resolve().parents[2]
                                           / "data" / "ninsheetmusic"))  # scores, metadata, processed/
    OUTPUT_DIR =BASE_DIR / "processed" / "annotation" / "annotations"
    ABC_DIR = BASE_DIR / "processed" / "rabc"

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process all ABC files, sorted for deterministic ordering
    all_files = sorted(ABC_DIR.glob("*.abc"))
    total = len(all_files)
    print(f"Found {total} ABC files in {ABC_DIR}")

    skipped = 0
    success = 0
    failed = 0

    for file_path in all_files:
        file_id = pathlib.Path(file_path).stem
        save_path = OUTPUT_DIR / f"{file_id}.json"
        processed = skipped + success + failed

        # Skip if already annotated
        if save_path.exists():
            skipped += 1
            print(f"[{processed+1}/{total}] Skipping {file_id} (already done)")
            continue

        print(f"[{processed+1}/{total}] Calling API for {file_id}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                abc_content = f.read()
            full_response = generate(file_id, abc_content, system_instruction_text)
        except Exception as e:
            print(f"  Error: {e}")
            raw_path = OUTPUT_DIR / f"{file_id}.raw.txt"
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(str(e))
            failed += 1
            continue

        # Parse and save the JSON response or raw text if parsing fails
        if full_response.strip():
            try:
                structured_data = json.loads(full_response)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(structured_data, f, indent=2)
                print(f"  Saved -> {save_path.name}")
                success += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: JSON parse failed: {e}")
                raw_path = OUTPUT_DIR / f"{file_id}.raw.txt"
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(full_response)
                print(f"  Saved raw -> {raw_path.name}")
                failed += 1

    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total:    {total}")
    print(f"  Skipped:  {skipped}  (already had .json)")
    print(f"  Success:  {success}  (new .json saved this run)")
    print(f"  Failed:   {failed}  (saved as .raw.txt)")
    print(f"  Remaining: {total - skipped - success - failed}")
