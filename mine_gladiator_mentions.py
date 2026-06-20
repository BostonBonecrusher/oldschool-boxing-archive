"""
mine_gladiator_mentions.py
---------------------------
Diagnostic tool — NOT the final search app. Jon noticed the live archive
only surfaces Commodus when asked about famous gladiators. This script
checks two things against the existing HistoricalWarfareChunk collection
(93,809 chunks, 60 source books — general Roman/Byzantine/Ottoman/Arab
military & political history, not a dedicated gladiator/arena history):

  1. EXACT NAME SWEEP — for a list of known gladiators/Servile-War figures,
     count exactly how many chunks contain that name anywhere in the text
     (via a `content LIKE *name*` filter — no relevance ranking, no
     semantic guessing, just "does this string appear at all"). This tells
     us definitively what's buried in the existing 60 books vs. genuinely
     absent.

  2. BROAD SEMANTIC SWEEP — a handful of hybrid (semantic + keyword)
     queries about gladiators/the arena in general, to see what the
     existing search ranking actually surfaces today and whether it's
     just a ranking problem (relevant content exists but loses to
     Commodus-heavy passages) or a true content gap.

Usage:
    python mine_gladiator_mentions.py
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
    "is_primary_source", "era", "civilization_or_nation",
    "conflict_focus", "ocr_used", "chunk_index", "page_number",
]

# ── Names to sweep for ──────────────────────────────────────────────
# Mix of: the Third Servile War leaders, famous named arena gladiators
# known from ancient sources/inscriptions, and Commodus as a control
# (we already know he shows up — confirms the sweep mechanism works).
NAME_SWEEP = [
    "Spartacus",
    "Crixus",
    "Oenomaus",
    "Castus",
    "Gannicus",
    "Commodus",      # control — known to already surface
    "Flamma",
    "Priscus",
    "Verus",
    "Carpophorus",
    "Tetraites",
    "Triumphus",
    "gladiator",      # generic term — shows total surface area of the topic
    "amphitheatre",
    "amphitheater",
    "Colosseum",
]

# ── Broad semantic queries ──────────────────────────────────────────
SEMANTIC_QUERIES = [
    "famous gladiators of ancient Rome",
    "gladiatorial combat in the arena",
    "training and life of gladiators",
    "the Third Servile War and the slave revolt of Spartacus",
    "famous warriors and duels in ancient history",
]


def main():
    print("Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )
    collection = client.collections.get(COLLECTION_NAME)

    total = collection.aggregate.over_all(total_count=True).total_count
    print(f"Connected. {COLLECTION_NAME} contains {total:,} chunks total.\n")

    # ═══════════════════════════════════════════════════════════════
    # PART 1 — EXACT NAME SWEEP
    # ═══════════════════════════════════════════════════════════════
    print("=" * 72)
    print("PART 1 — EXACT NAME SWEEP (content LIKE *name*, no ranking)")
    print("=" * 72)
    print("This counts every chunk where the name appears verbatim,")
    print("regardless of relevance. Tells us what's actually in the corpus.\n")

    found_any = {}

    for name in NAME_SWEEP:
        f = Filter.by_property("content").like(f"*{name}*")
        count = collection.aggregate.over_all(filters=f, total_count=True).total_count
        found_any[name] = count
        marker = "✅" if count > 0 else "❌"
        print(f"  {marker} {name:<16} {count:>4} chunk(s)")

    print("\n--- Sample chunks for every name that has hits (max 2 each) ---\n")

    for name, count in found_any.items():
        if count == 0 or name in ("gladiator", "amphitheatre", "amphitheater", "Colosseum"):
            continue  # skip generic terms here, too many hits to be useful as "samples"
        f = Filter.by_property("content").like(f"*{name}*")
        results = collection.query.fetch_objects(
            filters=f, limit=2, return_properties=RETURN_PROPS
        ).objects
        print(f"### {name} ({count} chunk(s) total) ###")
        for obj in results:
            p = obj.properties
            title = p.get("title") or p.get("source_file", "Unknown")
            year = p.get("year_published", 0)
            primary = " [PRIMARY SOURCE]" if p.get("is_primary_source") else ""
            content = p.get("content", "")
            preview = content[:350].replace("\n", " ") + ("..." if len(content) > 350 else "")
            print(f"  - {title}{f' ({year})' if year else ''}{primary}")
            print(f"    source_file: {p.get('source_file', '')}  chunk: {p.get('chunk_index', '?')}")
            print(f'    "{preview}"\n')
        print("-" * 72)

    # ═══════════════════════════════════════════════════════════════
    # PART 2 — BROAD SEMANTIC SWEEP
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("PART 2 — BROAD SEMANTIC SWEEP (hybrid search, top 5 each)")
    print("=" * 72)
    print("Shows what the current search ranking actually surfaces today")
    print("for general gladiator/warrior questions.\n")

    for query in SEMANTIC_QUERIES:
        print(f"\n### Query: \"{query}\"")
        results = collection.query.hybrid(
            query=query,
            limit=5,
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
            year = p.get("year_published", 0)
            content = p.get("content", "")
            preview = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")
            print(f"  [{i}] score={score:.3f}  {title}{f' ({year})' if year else ''}")
            print(f'      "{preview}"')
        print("-" * 72)

    client.close()

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    hits = {k: v for k, v in found_any.items() if v > 0}
    misses = [k for k, v in found_any.items() if v == 0]
    print(f"Names/terms with at least one hit: {', '.join(hits.keys()) if hits else '(none)'}")
    print(f"Names/terms with ZERO hits:        {', '.join(misses) if misses else '(none)'}")
    print("\nIf most named gladiators show 0 hits, this confirms the gap is in")
    print("the SOURCE LIBRARY, not the search logic — there simply isn't enough")
    print("dedicated gladiator/arena material in the 60 ingested books to talk")
    print("about anyone beyond Commodus. See the acquisition list doc for")
    print("specific public-domain books to add that would close this gap.")


if __name__ == "__main__":
    main()
