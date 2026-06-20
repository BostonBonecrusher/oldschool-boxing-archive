"""
mine_story_content.py
-----------------------
Content-mining tool (NOT a diagnostic, NOT the search app) -- pulls
candidate "interesting story" material from HistoricalWarfareChunk across
three buckets: gladiators, warriors, and legions. The goal is to surface
real, citable passages with enough narrative color to turn into Instagram
content (carousel/caption/reel), not just factual answers.

Unlike mine_gladiator_mentions.py (which checked whether a topic exists at
all in the corpus), this script assumes the topic exists and is hunting
specifically for the most VIVID, QUOTABLE, STORY-SHAPED passages: named
individuals, dramatic incidents, duels, mutinies, last stands, customs and
rituals, famous nicknames -- the stuff that makes a good carousel hook.

Writes full results to story_mining_results.md in this same folder (in
addition to printing to console) so they can be read back without needing
to paste terminal output by hand.

Usage:
    python mine_story_content.py
    python mine_story_content.py --per-query 8   # more candidates per query
"""

import os
import argparse
from datetime import datetime
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, HybridFusion

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "HistoricalWarfareChunk"
OUTPUT_FILE = "story_mining_results.md"

RETURN_PROPS = [
    "content", "source_file", "title", "author", "year_published",
    "is_primary_source", "era", "civilization_or_nation",
    "conflict_focus", "chunk_index", "page_number",
]

# ── Query buckets ────────────────────────────────────────────────────
# Each tuple is (bucket label, query text). Queries are written to surface
# narrative/anecdotal passages specifically, not generic overviews.
BUCKETS = {
    "GLADIATORS": [
        "a dramatic gladiator duel in the arena",
        "the training and daily life of gladiators in the ludus",
        "Spartacus and the Third Servile War revolt",
        "Commodus fighting personally as a gladiator in the arena",
        "the crowd's reaction to a gladiator's death or mercy",
    ],
    "WARRIORS": [
        "a legendary individual feat of bravery in battle",
        "single combat between champions before a battle",
        "a warrior famous for skill or ferocity in combat",
        "a dramatic last stand or heroic defeat",
        "a warrior's ritual or custom before going into battle",
    ],
    "LEGIONS": [
        "a famous Roman legion's nickname or tradition",
        "the legion's eagle standard and what its loss meant",
        "a mutiny or rebellion by soldiers against their commander",
        "the daily training and discipline of Roman soldiers",
        "decimation or harsh punishment in the Roman army",
        "a legion's famous siege or desperate defense",
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=6,
                        help="How many candidate chunks to pull per query (default 6)")
    parser.add_argument("--preview-chars", type=int, default=600,
                        help="How many characters of each chunk to show (default 600)")
    args = parser.parse_args()

    print("Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )
    collection = client.collections.get(COLLECTION_NAME)
    total = collection.aggregate.over_all(total_count=True).total_count
    print(f"Connected. {COLLECTION_NAME} contains {total:,} chunks total.\n")

    seen_uuids = {}  # uuid -> list of (bucket, query) it matched
    ordered_results = []  # list of (bucket, query, obj) in first-seen order

    lines = []
    lines.append(f"# Story Mining Results — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("Candidate passages for gladiator/warrior/legion Instagram content, "
                 "pulled via hybrid search against HistoricalWarfareChunk. Each entry "
                 "has the full source citation -- pick the ones with the best story "
                 "potential and cite the book + author + year in the actual content.")
    lines.append("")

    for bucket, queries in BUCKETS.items():
        print("=" * 72)
        print(f"BUCKET: {bucket}")
        print("=" * 72)
        lines.append(f"## {bucket}")
        lines.append("")

        for query in queries:
            print(f"\n### Query: \"{query}\"")
            lines.append(f"### Query: \"{query}\"")
            lines.append("")
            results = collection.query.hybrid(
                query=query,
                limit=args.per_query,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
                return_properties=RETURN_PROPS,
            ).objects

            if not results:
                print("  (no results)")
                lines.append("_(no results)_\n")
                continue

            for i, obj in enumerate(results, 1):
                uid = str(obj.uuid)
                p = obj.properties
                score = obj.metadata.score if obj.metadata else 0
                title = p.get("title") or p.get("source_file", "Unknown")
                author = p.get("author", "")
                year = p.get("year_published", 0)
                primary = p.get("is_primary_source", False)
                source_file = p.get("source_file", "")
                chunk_idx = p.get("chunk_index", "?")
                content = (p.get("content", "") or "").replace("\n", " ").strip()
                preview = content[:args.preview_chars]
                if len(content) > args.preview_chars:
                    preview += "..."

                dup_note = ""
                if uid in seen_uuids:
                    seen_uuids[uid].append((bucket, query))
                    dup_note = "  [also matched: " + "; ".join(
                        f"{b}/{q}" for b, q in seen_uuids[uid][:-1]) + "]"
                else:
                    seen_uuids[uid] = [(bucket, query)]
                    ordered_results.append((bucket, query, obj))

                cite = f"{title}"
                if author:
                    cite += f" — {author}"
                if year:
                    cite += f" ({year})"
                primary_tag = " [PRIMARY SOURCE]" if primary else ""

                print(f"  [{i}] score={score:.3f}  {cite}{primary_tag}{dup_note}")
                print(f'      "{preview}"')

                lines.append(f"**[{i}] score={score:.3f}** — {cite}{primary_tag}{dup_note}  ")
                lines.append(f"`source_file: {source_file}` | `chunk_index: {chunk_idx}`  ")
                lines.append(f"> {preview}")
                lines.append("")

            print("-" * 72)
            lines.append("---")
            lines.append("")

    client.close()

    unique_count = len(ordered_results)
    dup_count = sum(len(v) - 1 for v in seen_uuids.values())
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{unique_count} unique candidate chunks surfaced across all queries "
          f"({dup_count} cross-bucket repeats, which usually means a strong passage).")
    print(f"Full results written to {OUTPUT_FILE}")

    summary_lines = [
        "## Summary",
        "",
        f"{unique_count} unique candidate chunks surfaced across all queries "
        f"({dup_count} cross-bucket repeats -- a repeat across multiple queries "
        "usually flags an especially strong, versatile passage).",
        "",
    ]
    lines = lines[:2] + summary_lines + lines[2:]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
