"""
test_warfare_queries.py
------------------------
Quick sanity-check tool for the HistoricalWarfareChunk collection — run a
handful of representative hybrid searches and print the results (citation,
metadata tags, relevance score, content preview) so you can eyeball whether
chunking, OCR quality, and metadata tagging came out usable before building
the full search/Q&A app on top of it.

This is NOT the final search app — just a diagnostic. Run it once after
ingestion to spot-check quality across different source types (clean text
extraction vs. OCR, primary source vs. modern scholarship, etc.)

Usage:
    python test_warfare_queries.py
"""

import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "HistoricalWarfareChunk"

RETURN_PROPS = [
    "content", "source_file", "title", "author", "year_published",
    "is_primary_source", "document_type", "era", "time_period",
    "civilization_or_nation", "conflict_focus", "subject_commander",
    "ocr_used", "page_number", "chunk_index",
]

# A spread of test queries chosen to probe different things:
TEST_QUERIES = [
    {
        "label": "General training/combat topic (core scope check)",
        "query": "how did Roman soldiers train for combat",
    },
    {
        "label": "Soldier diet/nutrition (explicit scope item from Jon's brief)",
        "query": "what did soldiers eat on campaign, rations and diet",
    },
    {
        "label": "Specific commander/battle (tests commanders_mentioned/battles tagging)",
        "query": "Hannibal's tactics against the Romans",
    },
    {
        "label": "OCR quality check — targets the Adam Ferguson 1783 book specifically "
                  "(this is the file that previously crashed during ingestion)",
        "query": "the Roman Republic's decline and the causes of its fall",
    },
    {
        "label": "Primary-source / old-world perspective check",
        "query": "the discipline and conditioning of an army on the march",
    },
]


def main():
    print("Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )
    collection = client.collections.get(COLLECTION_NAME)

    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"Connected. {COLLECTION_NAME} contains {count:,} chunks total.\n")
    print("=" * 70)

    for test in TEST_QUERIES:
        print(f"\n### {test['label']}")
        print(f"Query: \"{test['query']}\"\n")

        results = collection.query.hybrid(
            query=test["query"],
            limit=3,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True),
            return_properties=RETURN_PROPS,
        ).objects

        if not results:
            print("  (no results)")
            continue

        for i, obj in enumerate(results, 1):
            p = obj.properties
            score = obj.metadata.score if obj.metadata else 0
            title = p.get("title") or p.get("source_file", "Unknown")
            author = p.get("author", "")
            year = p.get("year_published", 0)
            primary = " [PRIMARY SOURCE]" if p.get("is_primary_source") else ""
            ocr = " [OCR]" if p.get("ocr_used") else ""
            era = p.get("era", "")
            civ = p.get("civilization_or_nation", "")
            conflict = p.get("conflict_focus", "")
            commander = p.get("subject_commander", "")
            content = p.get("content", "")
            preview = content[:300].replace("\n", " ") + ("..." if len(content) > 300 else "")

            print(f"  [{i}] score={score:.3f}  {title}"
                  f"{f' by {author}' if author else ''}{f' ({year})' if year else ''}"
                  f"{primary}{ocr}")
            tags = [t for t in [era, civ, conflict, commander] if t]
            if tags:
                print(f"      tags: {' | '.join(tags)}")
            print(f"      source_file: {p.get('source_file', '')}  "
                  f"chunk: {p.get('chunk_index', '?')}")
            print(f'      "{preview}"\n')

        print("-" * 70)

    # ── Direct filtered check on the Adam Ferguson book ──────────────
    # None of the semantic queries above are guaranteed to surface this
    # specific file (Gibbon's "Decline and Fall" tends to outrank it on
    # decline/fall-themed queries since the title matches more closely).
    # This bypasses relevance ranking entirely and just pulls N chunks
    # straight from that file by source_file filter, so we can verify the
    # OCR text quality on the exact book that crashed the original
    # ingestion run, regardless of how well it ranks for any query.
    print("\n" + "=" * 70)
    print("### Direct check: Adam Ferguson (1783) chunks — bypasses relevance "
          "ranking, pulls straight from this file by source_file filter")
    ferguson_file = "AdamFerguson-TheHistoryOfTheProgressAndTerminationOfTheRomanRepublic-Complete1783.pdf"
    results = collection.query.fetch_objects(
        filters=Filter.by_property("source_file").equal(ferguson_file),
        limit=5,
        return_properties=RETURN_PROPS,
    ).objects

    if not results:
        print(f"  ⚠️  No chunks found for source_file == {ferguson_file}")
        print("      (Check the exact filename matches what's in the collection.)")
    else:
        print(f"  Found chunks from this file. Showing {len(results)}:\n")
        for i, obj in enumerate(results, 1):
            p = obj.properties
            ocr = " [OCR]" if p.get("ocr_used") else " [NOT flagged as OCR]"
            content = p.get("content", "")
            preview = content[:400].replace("\n", " ") + ("..." if len(content) > 400 else "")
            print(f"  [{i}]{ocr}  chunk_index={p.get('chunk_index', '?')}  "
                  f"page={p.get('page_number', '?')}")
            print(f'      "{preview}"\n')

    client.close()
    print("\nDone. Review above for: chunk readability, correct "
          "author/year/era tagging, and whether OCR'd chunks (esp. the "
          "Adam Ferguson 1783 book) read as usable text vs. garbled.")


if __name__ == "__main__":
    main()
