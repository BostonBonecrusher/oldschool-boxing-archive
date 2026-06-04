"""
01_create_schema.py
-------------------
Run this ONCE to set up the Weaviate collection (database structure).
After running successfully, you don't need to run it again unless you
want to delete everything and start over.

Usage:
    python 01_create_schema.py
"""

import os
import sys
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType, VectorDistances

load_dotenv()

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")

COLLECTION_NAME = "BoxingChunk"


def check_env():
    missing = []
    if not WEAVIATE_URL or "YOUR-CLUSTER" in WEAVIATE_URL:
        missing.append("WEAVIATE_URL")
    if not WEAVIATE_API_KEY or "your-weaviate" in WEAVIATE_API_KEY:
        missing.append("WEAVIATE_API_KEY")
    if not OPENAI_API_KEY or "your-openai" in OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        print("\n❌ ERROR: Missing or unfilled values in your .env file:")
        for var in missing:
            print(f"   - {var}")
        print("\nOpen your .env file and fill in those values, then run this script again.")
        sys.exit(1)


def main():
    check_env()

    print(f"\n🔌 Connecting to Weaviate Cloud at {WEAVIATE_URL}...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
        )
    except Exception as e:
        print(f"\n❌ Could not connect to Weaviate: {e}")
        print("\nThings to check:")
        print("  1. Is your WEAVIATE_URL correct? (no trailing slash)")
        print("  2. Is your WEAVIATE_API_KEY correct?")
        print("  3. Is your cluster awake? Log into console.weaviate.cloud and check.")
        sys.exit(1)

    print("✅ Connected!")

    if client.collections.exists(COLLECTION_NAME):
        answer = input(
            f"\n⚠️  Collection '{COLLECTION_NAME}' already exists.\n"
            "   Type 'delete' to delete it and start fresh, or press Enter to keep it: "
        ).strip().lower()
        if answer == "delete":
            client.collections.delete(COLLECTION_NAME)
            print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}'.")
        else:
            print("👍 Keeping existing collection. No changes made.")
            client.close()
            return

    print(f"\n🏗️  Creating collection '{COLLECTION_NAME}' with full historian metadata...")

    try:
        client.collections.create(
            name=COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                model="text-embedding-3-small",
            ),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[

                # ════════════════════════════════════════
                # CORE CONTENT
                # ════════════════════════════════════════

                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The text content of this chunk",
                ),
                Property(
                    name="source_file",
                    data_type=DataType.TEXT,
                    description="Original PDF filename",
                    skip_vectorization=True,
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Human-readable document title",
                ),

                # ════════════════════════════════════════
                # AUTHORSHIP & PUBLICATION
                # ════════════════════════════════════════

                Property(
                    name="author",
                    data_type=DataType.TEXT,
                    description="Author name, extracted from filename where possible",
                    skip_vectorization=True,
                ),
                Property(
                    name="year_published",
                    data_type=DataType.INT,
                    description="Year of publication from filename, or 0 if unknown",
                    skip_vectorization=True,
                ),
                Property(
                    name="decade_focus",
                    data_type=DataType.TEXT,
                    description="Decade the document covers, e.g. '1820s', '1950s', 'unknown'",
                    skip_vectorization=True,
                ),
                Property(
                    name="volume_number",
                    data_type=DataType.INT,
                    description="Volume number for multi-volume works (0 if not applicable)",
                    skip_vectorization=True,
                ),
                Property(
                    name="is_primary_source",
                    data_type=DataType.BOOL,
                    description="True if written at the time of the events (eyewitness/contemporary)",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # DOCUMENT CLASSIFICATION
                # ════════════════════════════════════════

                Property(
                    name="document_type",
                    data_type=DataType.TEXT,
                    description="historical_chronicle | training_manual | fight_account | biography | combat_manual | martial_arts | periodical | general",
                    skip_vectorization=True,
                ),
                Property(
                    name="discipline",
                    data_type=DataType.TEXT,
                    description="boxing | bare_knuckle | martial_arts | self_defense | mixed",
                    skip_vectorization=True,
                ),
                Property(
                    name="rules_era",
                    data_type=DataType.TEXT,
                    description="london_prize_ring (pre-1867) | queensberry_transition (1867-1900) | modern_queensberry (1900+) | unknown",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # HISTORICAL ERA
                # ════════════════════════════════════════

                Property(
                    name="era",
                    data_type=DataType.TEXT,
                    description="bare_knuckle_era (pre-1900) | golden_age (1900-1950) | midcentury (1950-1980) | modern (1980+) | unknown",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # FIGHTER & GEOGRAPHY
                # ════════════════════════════════════════

                Property(
                    name="subject_fighter",
                    data_type=DataType.TEXT,
                    description="Primary fighter(s) this document is about, from filename",
                    skip_vectorization=True,
                ),
                Property(
                    name="fighters_in_title",
                    data_type=DataType.TEXT,
                    description="Comma-separated fighter names found in the filename",
                    skip_vectorization=True,
                ),
                Property(
                    name="geographic_focus",
                    data_type=DataType.TEXT,
                    description="uk | usa | international | unknown",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # PHASE 2 — AI-TAGGED AFTER INGESTION
                # These are blank now. A separate script fills them in.
                # ════════════════════════════════════════

                Property(
                    name="weight_class_focus",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Weight class(es) discussed: heavyweight, middleweight, etc.",
                    skip_vectorization=True,
                ),
                Property(
                    name="fighters_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated fighter names found in this chunk's text",
                    skip_vectorization=True,
                ),
                Property(
                    name="venues_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated venues mentioned in this chunk",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_training_methods",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk describes training techniques or methods",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_fight_account",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk narrates an actual bout or fight",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_biographical_info",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk contains life story / biographical detail",
                    skip_vectorization=True,
                ),
                Property(
                    name="has_statistics",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk contains fight records, stats, or results",
                    skip_vectorization=True,
                ),
                Property(
                    name="quotes_present",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk contains a direct quote from a fighter, trainer, or eyewitness",
                    skip_vectorization=True,
                ),
                Property(
                    name="controversy_present",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk describes a disputed result, controversial decision, or scandal",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # CHUNK POSITION & TECHNICAL
                # ════════════════════════════════════════

                Property(
                    name="page_number",
                    data_type=DataType.INT,
                    description="Page number in source PDF where this chunk starts",
                    skip_vectorization=True,
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Sequential chunk number within the document",
                    skip_vectorization=True,
                ),
                Property(
                    name="total_pages",
                    data_type=DataType.INT,
                    description="Total pages in source PDF",
                    skip_vectorization=True,
                ),
                Property(
                    name="ocr_used",
                    data_type=DataType.BOOL,
                    description="True if OCR was used to extract this text",
                    skip_vectorization=True,
                ),

            ],
        )

        print(f"\n✅ Collection '{COLLECTION_NAME}' created successfully!")
        print("\n── PHASE 1 METADATA (auto-tagged during ingestion) ──────────")
        print("  content               — the text chunk itself")
        print("  source_file           — original PDF filename")
        print("  title                 — human-readable title")
        print("  author                — extracted from filename")
        print("  year_published        — year from filename")
        print("  decade_focus          — e.g. 1820s, 1950s")
        print("  volume_number         — for multi-volume works")
        print("  is_primary_source     — written at the time?")
        print("  document_type         — chronicle, biography, training_manual, etc.")
        print("  discipline            — boxing, bare_knuckle, martial_arts, etc.")
        print("  rules_era             — london_prize_ring, queensberry_transition, modern")
        print("  era                   — bare_knuckle_era, golden_age, midcentury, modern")
        print("  subject_fighter       — primary fighter from filename")
        print("  fighters_in_title     — all fighter names in filename")
        print("  geographic_focus      — uk, usa, international")
        print("  page_number / chunk_index / total_pages / ocr_used")
        print("\n── PHASE 2 METADATA (AI-tagged later, blank for now) ────────")
        print("  weight_class_focus    — heavyweight, middleweight, etc.")
        print("  fighters_mentioned    — fighter names found in chunk text")
        print("  venues_mentioned      — arenas/venues in chunk text")
        print("  contains_training_methods")
        print("  contains_fight_account")
        print("  contains_biographical_info")
        print("  has_statistics")
        print("\n👉 Next step: run  python 02_ingest.py")

    except Exception as e:
        print(f"\n❌ Failed to create collection: {e}")
        print("\nIf you see 'text2vec-openai not enabled', go to your Weaviate Cloud")
        print("dashboard and enable the OpenAI integration module.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
