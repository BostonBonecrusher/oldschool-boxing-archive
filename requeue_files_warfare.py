"""
requeue_files_warfare.py
--------------------------
Helper for re-ingesting a specific file (or files) cleanly after a fix to
the extraction pipeline -- e.g. the x_tolerance fix for jammed words
(The Cambridge Ancient History volumes) or the reversed-header fix.

02_ingest_warfare.py is resumable: it skips any filename already marked
"completed" in ingestion_log_warfare.json, and it never deletes existing
chunks before adding new ones. So to cleanly re-process a file you must:
  1. Delete that file's existing chunks from Weaviate (source_file match)
  2. Remove that filename from the "completed" list in the progress log

This script does both, for one or more filenames. After running it, just
run `python 02_ingest_warfare.py` as normal -- it will reprocess exactly
the files you requeued (using whatever extraction fixes are currently in
the script) and leave everything else untouched.

SAFE BY DEFAULT: dry run shows what would be deleted/requeued and makes no
changes. Pass --execute to actually do it.

Usage:
    python requeue_files_warfare.py "The Cambridge Ancient History, Vol. 12.pdf"
    python requeue_files_warfare.py "file1.pdf" "file2.pdf" --execute
"""

import os
import sys
import json
import argparse

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "HistoricalWarfareChunk"
LOG_FILE = "ingestion_log_warfare.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="+", help="Exact source_file name(s) to requeue, "
                         "matching the filename as it appears in the warfare_manifest.txt / "
                         "FILE_METADATA in 02_ingest_warfare.py")
    parser.add_argument("--execute", action="store_true",
                        help="Actually delete chunks + update the log (default: dry run)")
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

    for fname in args.filenames:
        agg = collection.aggregate.over_all(
            filters=Filter.by_property("source_file").equal(fname),
            total_count=True,
        )
        n = agg.total_count
        print(f"\n--- {fname} ---")
        print(f"   Existing chunks in Weaviate: {n:,}")
        if n == 0:
            print("   (no chunks found -- check the filename matches exactly, including "
                  "punctuation/spacing)")
            continue

        if not args.execute:
            print("   DRY RUN — would delete these chunks and clear from progress log.")
            continue

        result = collection.data.delete_many(
            where=Filter.by_property("source_file").equal(fname)
        )
        print(f"   Deleted: {result.successful} chunks "
              f"({result.failed} failed)" if hasattr(result, "successful") else f"   Deleted.")

    client.close()

    if not args.execute:
        print(f"\nDRY RUN — no changes made. Re-run with --execute to delete these chunks "
              f"and requeue the file(s) for re-ingestion.")
        return

    # Clear from progress log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {"completed": [], "failed": {}}

    removed = []
    for fname in args.filenames:
        if fname in log.get("completed", []):
            log["completed"].remove(fname)
            removed.append(fname)
        log.get("failed", {}).pop(fname, None)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"\nRemoved {len(removed)} filename(s) from {LOG_FILE}'s completed list.")
    print("Now run:  python 02_ingest_warfare.py")
    print("It will reprocess exactly the file(s) you just requeued and leave everything else alone.")


if __name__ == "__main__":
    main()
