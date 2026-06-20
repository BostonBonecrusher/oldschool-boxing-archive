"""
fix_reversed_headers_warfare.py
--------------------------------
One-time maintenance pass over the HistoricalWarfareChunk collection to fix
a pdfplumber extraction artifact: a recurring header/footer text element
(chapter title, photo credit line, etc.) sometimes gets extracted with its
ENTIRE character order reversed -- e.g. "cilbuP tnemniatretnE" instead of
"Public Entertainment". Confirmed via direct pdftotext extraction that the
source PDFs themselves are clean (e.g. TheHistoryOfAncientRome.pdf) -- this
is introduced by pdfplumber's own word/line clustering for that specific
text run, not a defect in the books.

This is a PURE, DETERMINISTIC string repair -- no LLM call, no API cost.
Detection: a run of 3+ consecutive tokens that each start lowercase and end
uppercase (e.g. "semaG", "cilbuP") essentially never happens in real prose,
but is exactly what falls out when a Title Case phrase is reversed
character-by-character. The fix reverses the matched span back to its
original character order.

This same fix has also been added to 02_ingest_warfare.py's clean_text(),
so any FUTURE/re-ingested file is protected automatically. This script
patches chunks that are ALREADY sitting in Weaviate from before that fix
existed.

SAFE BY DEFAULT: running this script with no flags only COUNTS matching
chunks and shows a breakdown by source file -- it makes no changes and
costs nothing (no OpenAI calls at all). Pass --execute to actually update.

Usage:
    python fix_reversed_headers_warfare.py              # dry run (count only)
    python fix_reversed_headers_warfare.py --execute     # actually fix + update
"""

import os
import sys
import json
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
LOG_FILE = "reversed_header_fix_log.json"
SCAN_PROPS = ["content", "source_file"]


# ─────────────────────────────────────────────────────────────
# Same detection/repair logic as 02_ingest_warfare.py -- kept in
# sync deliberately so this script and future ingestion runs
# agree on what counts as "reversed".
# ─────────────────────────────────────────────────────────────

def _is_reversed_token(tok):
    core = tok.strip(",.;:()-–—\"'")
    if len(core) < 3:
        return False
    if not core[0].isalpha() or not core[-1].isalpha():
        return False
    return core[0].islower() and core[-1].isupper()


def _is_bridgeable_glue(tok):
    core = tok.strip(",.;:()-–—\"'")
    if core == "":
        return True
    if core.isdigit():
        return True
    if core.isalpha() and core.isupper() and len(core) <= 5:
        return True
    return False


def fix_reversed_header_runs(text):
    if not text:
        return text
    tokens = text.split(' ')
    n = len(tokens)
    flagged = [_is_reversed_token(t) for t in tokens]
    i = 0
    out = []
    while i < n:
        if flagged[i]:
            j = i
            last_flagged = i
            flagged_count = 1
            while j + 1 < n:
                nxt = j + 1
                if flagged[nxt]:
                    last_flagged = nxt
                    flagged_count += 1
                    j = nxt
                elif _is_bridgeable_glue(tokens[nxt]):
                    j = nxt
                else:
                    break
            if flagged_count >= 3:
                span = tokens[i:last_flagged + 1]
                out.append(' '.join(span)[::-1])
                i = last_flagged + 1
                continue
        out.append(tokens[i])
        i += 1
    return ' '.join(out)


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": {}}


def save_log(log):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Actually perform the fix + updates (default: dry run, count only)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional cap on number of chunks to process this run (0 = no cap)")
    parser.add_argument("--show-samples", type=int, default=5,
                        help="How many before/after samples to print in dry-run mode (default 5)")
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
    print("✅ Connected. Scanning full collection for reversed-header artifacts...\n")

    matches = []   # list of (uuid, original, fixed)
    by_file = Counter()
    scanned = 0

    for obj in collection.iterator(return_properties=SCAN_PROPS):
        scanned += 1
        props = obj.properties
        content = props.get("content", "") or ""
        fixed = fix_reversed_header_runs(content)
        if fixed != content:
            matches.append((str(obj.uuid), content, fixed))
            by_file[props.get("source_file", "unknown")] += 1
        if scanned % 10000 == 0:
            print(f"   ...scanned {scanned:,} chunks so far")

    print(f"\nScanned {scanned:,} total chunks. {len(matches):,} contain a reversed-header artifact.\n")
    print("Breakdown by source file:")
    for fname, n in sorted(by_file.items(), key=lambda x: -x[1]):
        print(f"   {n:>6,}  {fname}")

    if not args.execute:
        print(f"\nSample fixes (showing up to {args.show_samples}):\n")
        for uuid, before, after in matches[:args.show_samples]:
            # show just the changed region for readability
            diff_idx = next((i for i, (a, b) in enumerate(zip(before, after)) if a != b), 0)
            print(f"  BEFORE: ...{before[max(0,diff_idx-40):diff_idx+100]}...")
            print(f"  AFTER : ...{after[max(0,diff_idx-40):diff_idx+100]}...")
            print()
        print(f"DRY RUN — no changes made. Re-run with --execute to fix these "
              f"{len(matches):,} chunks. This is a pure string repair (no OpenAI "
              "cost) but Weaviate will still re-embed each updated chunk.")
        client.close()
        return

    log = load_log()
    already_done = set(log["completed"]) | set(log["failed"].keys())
    todo = [(u, b, a) for u, b, a in matches if u not in already_done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"\n{len(already_done):,} chunks already processed in a previous run "
          f"(resuming from {LOG_FILE}). {len(todo):,} chunks to process now.\n")

    if not todo:
        print("Nothing to do.")
        client.close()
        return

    done_count = 0
    fail_count = 0
    for i, (uuid, before, after) in enumerate(todo, 1):
        try:
            collection.data.update(uuid=uuid, properties={"content": after})
            log["completed"].append(uuid)
            done_count += 1
        except Exception as e:
            log["failed"][uuid] = str(e)
            fail_count += 1
        if i % 50 == 0 or i == len(todo):
            print(f"   {i:,}/{len(todo):,} processed ({done_count:,} fixed, {fail_count:,} failed)")
            save_log(log)

    save_log(log)
    client.close()

    print("\n" + "=" * 60)
    print(f"Done. Fixed: {done_count:,}  Failed: {fail_count:,}")
    if fail_count:
        print(f"Failures are recorded in {LOG_FILE} under \"failed\" — re-run the "
              "script with --execute again to retry just those.")


if __name__ == "__main__":
    main()
