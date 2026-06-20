"""
scan_jammed_words_warfare.py
------------------------------
Read-only diagnostic over the HistoricalWarfareChunk collection to find a
text-quality defect: chunks where words run together with NO spaces at
all, e.g. "defiantChristiansarenotautomaticallypunishedwithdeath" instead
of "defiant Christians are not automatically punished with death."

Root cause (confirmed against The Cambridge Ancient History, Vol. 12):
pdfplumber's extract_text() default x_tolerance (3pt) is too generous for
this PDF's font/kerning, so the gap between two adjacent words gets read
as still "inside" one word. Lowering x_tolerance to 1.5 fixes it cleanly
(verified directly against the source PDF, and verified it does NOT
introduce false word-splits on this or several other already-clean books).

That fix is already in 02_ingest_warfare.py's extract_text_pdfplumber().
This script does NOT try to repair already-jammed chunk text in place
(re-inserting spaces into already-corrupted text is a lossy guessing
game). Instead it tells you exactly which files/chunks are affected, so
you can re-ingest just those files cleanly from the source PDF using
requeue_files_warfare.py + a normal `python 02_ingest_warfare.py` run.

This is read-only -- it never modifies Weaviate. Safe to run anytime.

Usage:
    python scan_jammed_words_warfare.py
    python scan_jammed_words_warfare.py --min-run 25   # tune sensitivity
"""

import os
import re
import sys
import argparse
from collections import Counter

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "HistoricalWarfareChunk"
SCAN_PROPS = ["content", "source_file", "chunk_index"]


def jammed_run_length(content, min_run):
    """Returns the length of the longest unbroken run of letters found in
    the chunk (no spaces/punctuation). A run this long basically never
    occurs in real English/Latin prose -- it means several real words got
    fused together with no space."""
    longest = 0
    for run in re.findall(r"[A-Za-z]+", content):
        if len(run) > longest:
            longest = len(run)
    return longest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-run", type=int, default=25,
                        help="Flag a chunk if it contains an unbroken letter run at least "
                             "this long (default 25 -- real English words essentially never "
                             "exceed ~20 chars, e.g. 'incomprehensibility' is 20)")
    parser.add_argument("--show-samples", type=int, default=3,
                        help="How many sample chunks to print per affected file (default 3)")
    args = parser.parse_args()

    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        print("❌ Missing WEAVIATE_URL / WEAVIATE_API_KEY in .env file.")
        sys.exit(1)

    print("🔌 Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY} if OPENAI_API_KEY else None,
    )
    collection = client.collections.get(COLLECTION_NAME)
    print(f"✅ Connected. Scanning full collection for jammed-word chunks "
          f"(longest letter-run >= {args.min_run})...\n")

    by_file = Counter()
    samples_by_file = {}
    scanned = 0
    total_matches = 0

    for obj in collection.iterator(return_properties=SCAN_PROPS):
        scanned += 1
        props = obj.properties
        content = props.get("content", "") or ""
        run_len = jammed_run_length(content, args.min_run)
        if run_len >= args.min_run:
            fname = props.get("source_file", "unknown")
            by_file[fname] += 1
            total_matches += 1
            samples_by_file.setdefault(fname, [])
            if len(samples_by_file[fname]) < args.show_samples:
                samples_by_file[fname].append((str(obj.uuid), content[:200]))
        if scanned % 10000 == 0:
            print(f"   ...scanned {scanned:,} chunks so far")

    client.close()

    print(f"\nScanned {scanned:,} total chunks. {total_matches:,} look jammed.\n")
    print("Breakdown by source file (this is your re-ingestion list):")
    for fname, n in sorted(by_file.items(), key=lambda x: -x[1]):
        print(f"   {n:>6,}  {fname}")

    print(f"\nSample jammed chunks (up to {args.show_samples} per file):\n")
    for fname, samples in samples_by_file.items():
        print(f"--- {fname} ---")
        for uuid, preview in samples:
            print(f"   [{uuid}] {preview}...")
        print()

    if by_file:
        file_list = " ".join(f'"{f}"' for f in by_file.keys())
        print("To fix: re-ingest these specific files from the source PDF with the\n"
              "corrected pdfplumber x_tolerance (already patched into\n"
              "02_ingest_warfare.py's extract_text_pdfplumber). Run:\n")
        print(f"   python requeue_files_warfare.py {file_list}")
        print("   python 02_ingest_warfare.py")
        print("\n(requeue_files_warfare.py deletes that file's existing chunks and clears\n"
              "it from the progress log so the next ingestion run reprocesses it fresh.)")
    else:
        print("No jammed-word chunks found with this threshold.")


if __name__ == "__main__":
    main()
