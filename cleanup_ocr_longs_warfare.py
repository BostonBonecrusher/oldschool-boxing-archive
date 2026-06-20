"""
cleanup_ocr_longs_warfare.py
-----------------------------
One-time maintenance pass over the HistoricalWarfareChunk collection to fix
a specific, well-understood OCR artifact: pre-~1810 English printing used
the historical long-s character (ſ), which OCR engines (including our own
Tesseract pass, and whatever OCR was baked into some PDFs' pre-existing
text layers, e.g. Gibbon) consistently misread as "f" — e.g. "ftate" for
"state", "confifting" for "consisting".

This targets BOTH:
  - chunks we OCR'd ourselves (ocr_used = True)
  - chunks from books published <= 1810 even if we extracted their text
    layer directly via pdfplumber (the text layer itself can already carry
    this error if whoever digitized the PDF originally used a modern-
    English OCR model — confirmed on Gibbon's "Decline and Fall" volumes)

Cleanup is done with gpt-4o-mini, NOT a regex, because f-vs-s is genuinely
context-dependent (e.g. "for", "first", "from" must stay as-is). The
prompt is deliberately conservative: fix clear OCR character errors only,
never touch genuine period spelling/vocabulary (this archive's whole
value is the old-world voice — "publick", "encrease", "compleat" etc. are
NOT errors and must be preserved), never touch names/dates/numbers/quotes
beyond an obvious character-level misread, and leave anything too garbled
to confidently fix exactly as-is rather than guessing.

SAFE BY DEFAULT: running this script with no flags only COUNTS matching
chunks and shows a breakdown by source file — it makes no changes and
costs nothing. Pass --execute to actually run the cleanup.

Usage:
    python cleanup_ocr_longs_warfare.py              # dry run (count only)
    python cleanup_ocr_longs_warfare.py --execute     # actually clean + update
    python cleanup_ocr_longs_warfare.py --execute --workers 4   # fewer threads
"""

import os
import sys
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from openai import OpenAI

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "HistoricalWarfareChunk"
LOG_FILE = "ocr_cleanup_log.json"
YEAR_CUTOFF = 1810          # long-s had largely disappeared from English printing by ~1800-1810
CLEANUP_MODEL = "gpt-4o-mini"
SCAN_PROPS = ["content", "source_file", "ocr_used", "year_published"]

SYSTEM_PROMPT = """You are cleaning up OCR output from scanned historical books printed before \
~1810, when English typography used the long-s character (ſ), which OCR software consistently \
misreads as "f". Your ONLY job is to fix clear OCR character-recognition errors. Do NOT \
modernize spelling, rephrase, summarize, add content, or remove content.

FIX these OCR artifacts when you see them:
- Long-s misread as f: e.g. "ftate" -> "state", "confifting" -> "consisting", "fubject" -> "subject"
- Obviously broken/garbled character sequences from poor scan quality (stray symbols, a letter \
dropped into the middle of a word, badly split words) — restore the most likely original word \
ONLY if you are highly confident from context; otherwise leave it exactly as-is.
- Other common old-typeface OCR letter confusions (e.g. rn/m) only when the result is obviously \
nonsensical and the fix is clear from context.

DO NOT touch:
- Genuine period-correct spelling/vocabulary that is NOT an OCR error — e.g. "publick", \
"encrease", "compleat", "antient", "chymistry" are authentic period spellings, not mistakes. \
This is a historical primary-source archive; preserving the original old-world voice matters. \
You are removing OCR noise, not modernizing the prose.
- Any name, number, date, or direct quote — preserve exactly, unless it is an obvious \
character-level OCR misread (never change what the name/number/date actually IS).
- Sentence structure, word choice, or meaning — never paraphrase or rewrite.

If a passage is too garbled to confidently reconstruct, leave that part exactly as-is rather \
than guessing. If the text has no OCR errors at all, return it completely unchanged.

Return ONLY the corrected text. No commentary, no explanation, no markdown, no wrapping quotes."""


def needs_cleanup(props):
    if props.get("ocr_used"):
        return True
    year = props.get("year_published") or 0
    return 0 < year <= YEAR_CUTOFF


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_log(log):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


_log_lock = threading.Lock()


def clean_chunk_text(openai_client, text):
    """Returns cleaned text, or None on failure after retries."""
    for attempt in range(3):
        try:
            resp = openai_client.chat.completions.create(
                model=CLEANUP_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.0,
                max_tokens=2000,
            )
            cleaned = resp.choices[0].message.content
            if cleaned and cleaned.strip():
                return cleaned.strip()
        except Exception as e:
            print(f"      ⚠️  Cleanup attempt {attempt+1} failed: {e}")
    return None


def process_one(collection, openai_client, uuid, content, log, save_every_lock_counter):
    cleaned = clean_chunk_text(openai_client, content)
    if cleaned is None:
        with _log_lock:
            log["failed"][uuid] = "openai_call_failed"
        return False
    try:
        collection.data.update(uuid=uuid, properties={"content": cleaned})
    except Exception as e:
        with _log_lock:
            log["failed"][uuid] = f"weaviate_update_failed: {e}"
        return False
    with _log_lock:
        log["completed"].append(uuid)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform cleanup + updates (default: dry run, count only)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent worker threads (default: 8)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional cap on number of chunks to process this run (0 = no cap)")
    args = parser.parse_args()

    if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_API_KEY:
        print("❌ Missing WEAVIATE_URL / WEAVIATE_API_KEY / OPENAI_API_KEY in .env file.")
        sys.exit(1)

    print("🔌 Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )
    collection = client.collections.get(COLLECTION_NAME)
    print("✅ Connected. Scanning full collection for affected chunks "
          "(ocr_used=True OR year_published <= "
          f"{YEAR_CUTOFF})...\n")

    matches = []          # list of (uuid, content)
    by_file = Counter()
    scanned = 0

    for obj in collection.iterator(return_properties=SCAN_PROPS):
        scanned += 1
        props = obj.properties
        if needs_cleanup(props):
            matches.append((str(obj.uuid), props.get("content", "")))
            by_file[props.get("source_file", "unknown")] += 1
        if scanned % 10000 == 0:
            print(f"   ...scanned {scanned:,} chunks so far")

    print(f"\nScanned {scanned:,} total chunks. {len(matches):,} match the cleanup criteria.\n")
    print("Breakdown by source file:")
    for fname, n in sorted(by_file.items(), key=lambda x: -x[1]):
        print(f"   {n:>6,}  {fname}")

    if not args.execute:
        print(f"\nDRY RUN — no changes made. Re-run with --execute to clean these "
              f"{len(matches):,} chunks (uses {CLEANUP_MODEL}, costs a small amount of "
              "OpenAI API usage and will re-embed each updated chunk).")
        client.close()
        return

    log = load_log()
    already_done = set(log["completed"]) | set(log["failed"].keys())
    todo = [(u, c) for u, c in matches if u not in already_done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n{len(already_done):,} chunks already processed in a previous run "
          f"(resuming from {LOG_FILE}). {len(todo):,} chunks to process now.\n")

    if not todo:
        print("Nothing to do.")
        client.close()
        return

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    done_count = 0
    fail_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, collection, openai_client, uuid, content, log, None): uuid
            for uuid, content in todo
        }
        for i, future in enumerate(as_completed(futures), 1):
            ok = future.result()
            if ok:
                done_count += 1
            else:
                fail_count += 1
            if i % 25 == 0 or i == len(todo):
                print(f"   {i:,}/{len(todo):,} processed "
                      f"({done_count:,} cleaned, {fail_count:,} failed)")
                save_log(log)

    save_log(log)
    client.close()

    print("\n" + "=" * 60)
    print(f"Done. Cleaned: {done_count:,}  Failed: {fail_count:,}")
    if fail_count:
        print(f"Failures are recorded in {LOG_FILE} under \"failed\" — re-run the "
              "script with --execute again to retry just those (already-completed "
              "chunks are skipped automatically).")


if __name__ == "__main__":
    main()
