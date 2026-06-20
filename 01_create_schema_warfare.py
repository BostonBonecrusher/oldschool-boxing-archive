"""
01_create_schema_warfare.py
----------------------------
Run this ONCE to set up the Weaviate collection for the Historical Warfare
archive (Roman/Greek/Byzantine/Ottoman/Arab/conquest history — the "roman
empire" book folder). Same cluster as the boxing archive, separate collection.

After running successfully, you don't need to run it again unless you
want to delete everything and start over.

Usage:
    python 01_create_schema_warfare.py
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

COLLECTION_NAME = "HistoricalWarfareChunk"


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
                    description="Author name (verified where possible from filename or title page; blank if uncertain)",
                    skip_vectorization=True,
                ),
                Property(
                    name="year_published",
                    data_type=DataType.INT,
                    description="Year of this edition/translation/printing from filename, or 0 if unknown",
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
                    description="True if this is an ancient/classical text written by an author of that era (e.g. Tacitus, Plutarch, Josephus), not modern scholarship about that era",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # DOCUMENT CLASSIFICATION
                # ════════════════════════════════════════

                Property(
                    name="document_type",
                    data_type=DataType.TEXT,
                    description="primary_source_text | historical_chronicle | academic_article | biography | military_treatise | general",
                    skip_vectorization=True,
                ),
                Property(
                    name="discipline",
                    data_type=DataType.TEXT,
                    description="cavalry | military_medicine | religion_and_culture | diplomacy_and_policy | administration_and_discipline | campaign_history | tactics_and_strategy | leadership_and_command | siege_warfare | general_military_history",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # HISTORICAL ERA & CONTEXT
                # ════════════════════════════════════════

                Property(
                    name="era",
                    data_type=DataType.TEXT,
                    description="ancient (pre-500 CE) | byzantine (500-1453 CE) | ottoman (1300-1922 CE) | early_modern_conquest (1492-1700 CE, Americas) | modern (1700 CE+) | multi_era | unknown",
                    skip_vectorization=True,
                ),
                Property(
                    name="time_period",
                    data_type=DataType.TEXT,
                    description="Free-text, human-readable period covered, e.g. 'Roman Republic', 'High Roman Empire (70-192 CE)', 'Fall of Constantinople (1453 CE)'",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # CIVILIZATION, CONFLICT & GEOGRAPHY
                # ════════════════════════════════════════

                Property(
                    name="civilization_or_nation",
                    data_type=DataType.TEXT,
                    description="Comma-separated civilizations/nations covered, e.g. 'roman', 'byzantine, arab', 'spanish, inca'",
                    skip_vectorization=True,
                ),
                Property(
                    name="conflict_focus",
                    data_type=DataType.TEXT,
                    description="Specific named conflict or military topic this document centers on, e.g. punic_wars, fall_of_constantinople, roman_army_organization, spanish_conquest_of_peru",
                    skip_vectorization=True,
                ),
                Property(
                    name="geographic_focus",
                    data_type=DataType.TEXT,
                    description="mediterranean | middle_east | europe | americas | unknown",
                    skip_vectorization=True,
                ),
                Property(
                    name="subject_commander",
                    data_type=DataType.TEXT,
                    description="Primary commander/ruler/military figure this document is about, from filename or known-work table (comma-separated if more than one)",
                    skip_vectorization=True,
                ),

                # ════════════════════════════════════════
                # PHASE 2 — AI-TAGGED AFTER INGESTION
                # These are blank now. A separate script fills them in.
                # ════════════════════════════════════════

                Property(
                    name="commanders_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated commander/ruler names found in this chunk's text",
                    skip_vectorization=True,
                ),
                Property(
                    name="battles_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated named battles/sieges/campaigns mentioned in this chunk",
                    skip_vectorization=True,
                ),
                Property(
                    name="units_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated military units/formations mentioned (e.g. legion, phalanx, janissary)",
                    skip_vectorization=True,
                ),
                Property(
                    name="weapons_or_tech_mentioned",
                    data_type=DataType.TEXT,
                    description="[Phase 2] Comma-separated weapons, armor, or military technology mentioned in this chunk",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_battle_account",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk narrates an actual battle, siege, or engagement",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_tactical_analysis",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk discusses tactics, formations, or battlefield decision-making",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_strategic_overview",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk discusses broader strategy, campaign planning, or grand strategy",
                    skip_vectorization=True,
                ),
                Property(
                    name="contains_biographical_info",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk contains life story / biographical detail about a commander or figure",
                    skip_vectorization=True,
                ),
                Property(
                    name="has_statistics",
                    data_type=DataType.BOOL,
                    description="[Phase 2] True if chunk contains troop numbers, casualty figures, or other quantitative data",
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
        print("  author                — verified from filename/title page where possible")
        print("  year_published        — edition/printing year from filename")
        print("  volume_number         — for multi-volume works")
        print("  is_primary_source     — ancient/classical author, not modern scholarship")
        print("  document_type         — primary_source_text, historical_chronicle, etc.")
        print("  discipline            — cavalry, military_medicine, tactics_and_strategy, etc.")
        print("  era                   — ancient, byzantine, ottoman, early_modern_conquest, etc.")
        print("  time_period           — human-readable period covered")
        print("  civilization_or_nation— roman, greek, byzantine, ottoman, arab, etc.")
        print("  conflict_focus        — specific named conflict/topic")
        print("  geographic_focus      — mediterranean, middle_east, europe, americas")
        print("  subject_commander     — primary commander/figure")
        print("  page_number / chunk_index / total_pages / ocr_used")
        print("\n── PHASE 2 METADATA (AI-tagged later, blank for now) ────────")
        print("  commanders_mentioned, battles_mentioned, units_mentioned,")
        print("  weapons_or_tech_mentioned, contains_battle_account,")
        print("  contains_tactical_analysis, contains_strategic_overview,")
        print("  contains_biographical_info, has_statistics")
        print("\n👉 Next step: run  python 02_ingest_warfare.py")

    except Exception as e:
        print(f"\n❌ Failed to create collection: {e}")
        print("\nIf you see 'text2vec-openai not enabled', go to your Weaviate Cloud")
        print("dashboard and enable the OpenAI integration module.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
