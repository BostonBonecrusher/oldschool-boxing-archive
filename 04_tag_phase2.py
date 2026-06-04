"""
04_tag_phase2.py
----------------
Goes through every chunk already in your Weaviate archive and uses AI to
fill in the Phase 2 metadata fields that were left blank during ingestion:

    weight_class_focus        — heavyweight, middleweight, etc.
    fighters_mentioned        — every fighter name found in the chunk
    venues_mentioned          — arenas, stadiums, cities mentioned
    contains_training_methods — True if chunk describes training techniques
    contains_fight_account    — True if chunk narrates an actual bout
    contains_biographical_info— True if chunk contains life story details
    has_statistics            — True if chunk has fight records or stats

Uses gpt-4o-mini (fast + cheap — this is classification, not storytelling).
Saves progress after every chunk so you can stop and restart safely.

Usage:
    python 04_tag_phase2.py

To re-tag everything from scratch (wipes the progress log):
    python 04_tag_phase2.py --reset
"""

import os
import sys
import json
import time
import argparse
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from openai import OpenAI

load_dotenv()

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME  = "BoxingChunk"

# Progress log — stores UUIDs already tagged so we can resume
LOG_FILE = os.path.join(os.path.dirname(__file__), "tag_phase2_log.json")

# Pause between chunks (seconds) — keeps OpenAI rate limits happy
RATE_LIMIT_PAUSE = 0.15


# ─────────────────────────────────────────────────────────────
# PROGRESS LOG
# ─────────────────────────────────────────────────────────────

def load_log():
    """Load set of already-tagged UUIDs from the log file."""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
            return set(data.get("tagged_uuids", []))
    return set()


def save_log(tagged_uuids):
    """Save the set of tagged UUIDs to the log file."""
    with open(LOG_FILE, "w") as f:
        json.dump({"tagged_uuids": list(tagged_uuids)}, f)


# ─────────────────────────────────────────────────────────────
# AI TAGGING
# ─────────────────────────────────────────────────────────────

TAGGING_PROMPT = """\
You are a boxing history archivist. Read the following passage from a boxing book or document.
Your job is to tag it accurately for a search database.

Return ONLY valid JSON with exactly these fields — nothing else:

{
  "weight_class_focus": "<comma-separated weight classes mentioned, e.g. 'heavyweight, middleweight', or 'unknown' if none>",
  "fighters_mentioned": "<comma-separated fighter names found in the text, or 'none' if no fighters named>",
  "venues_mentioned": "<comma-separated venues, arenas, cities where fights took place, or 'none' if none>",
  "contains_training_methods": <true or false — true if the passage describes training techniques, drills, conditioning methods, or physical preparation>,
  "contains_fight_account": <true or false — true if the passage narrates or describes an actual fight or bout>,
  "contains_biographical_info": <true or false — true if the passage contains life story, personal history, or biographical detail about a fighter>,
  "has_statistics": <true or false — true if the passage contains fight records, win/loss figures, knockout counts, or numerical fight statistics>,
  "quotes_present": <true or false — true if the passage contains a direct quote (words spoken or written) from a fighter, trainer, manager, referee, journalist, or eyewitness — must be clearly attributed or presented as speech>,
  "controversy_present": <true or false — true if the passage describes or references a disputed result, controversial decision, accusation of foul play, scandal, robbery, or any disputed claim about a fight or fighter>
}

Rules:
- For fighters_mentioned: include ONLY clearly named fighters (first + last name or well-known single name like "Dempsey"). Do not guess or infer. Do not include trainers, managers, or promoters unless they also fought.
- For venues_mentioned: include named arenas, stadiums, clubs, and cities where fights specifically took place.
- For weight classes: use standard terms (heavyweight, light heavyweight, middleweight, welterweight, lightweight, featherweight, bantamweight, flyweight). Include any mentioned.
- For quotes_present: only mark True if there is a clear direct quote — words in quotation marks attributed to a real person, or clearly presented as direct speech. Paraphrases do not count.
- For controversy_present: mark True if there is any disputed result, accusations of bias, claims a fight was fixed, complaints about judging, or any contested historical claim about a bout or fighter's record/reputation.
- Be conservative on booleans — only mark True if the passage clearly and explicitly contains that type of content.

Passage to tag:
"""


def _safe_defaults():
    """Return blank-but-valid Phase 2 tags for chunks that can't be parsed."""
    return {
        "weight_class_focus":         "unknown",
        "fighters_mentioned":         "none",
        "venues_mentioned":           "none",
        "contains_training_methods":  False,
        "contains_fight_account":     False,
        "contains_biographical_info": False,
        "has_statistics":             False,
        "quotes_present":             False,
        "controversy_present":        False,
    }


def sanitize_content(content):
    """Strip control characters that can break JSON output."""
    import re
    # Remove control chars except tab and newline
    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', content)
    return content[:2000]


def tag_chunk(openai_client, content):
    """
    Send a chunk's text to gpt-4o-mini and get back Phase 2 tags as a dict.
    - JSON errors   → returns safe defaults (marks chunk done, no retry)
    - API errors    → returns None (chunk will be retried next run)
    """
    content_clean = sanitize_content(content)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": TAGGING_PROMPT + content_clean}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content
        tags = json.loads(raw)
        return tags
    except json.JSONDecodeError as e:
        # Model returned malformed JSON — use safe defaults and move on
        print(f"   ⚠️  JSON parse failed (using defaults): {e}")
        return _safe_defaults()
    except Exception as e:
        # API/connection error — return None so chunk retries next run
        print(f"   ⚠️  AI tagging failed: {e}")
        return None


def validate_tags(tags):
    """
    Make sure the AI returned something sensible.
    Fills in safe defaults for any missing or malformed fields.
    """
    safe = {
        "weight_class_focus":         str(tags.get("weight_class_focus", "unknown")),
        "fighters_mentioned":         str(tags.get("fighters_mentioned", "none")),
        "venues_mentioned":           str(tags.get("venues_mentioned", "none")),
        "contains_training_methods":  bool(tags.get("contains_training_methods", False)),
        "contains_fight_account":     bool(tags.get("contains_fight_account", False)),
        "contains_biographical_info": bool(tags.get("contains_biographical_info", False)),
        "has_statistics":             bool(tags.get("has_statistics", False)),
        "quotes_present":             bool(tags.get("quotes_present", False)),
        "controversy_present":        bool(tags.get("controversy_present", False)),
    }
    return safe


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe the progress log and re-tag everything from scratch"
    )
    args = parser.parse_args()

    # ── Check config ──
    missing = []
    if not WEAVIATE_URL or "YOUR-CLUSTER" in WEAVIATE_URL:
        missing.append("WEAVIATE_URL")
    if not WEAVIATE_API_KEY or "your-weaviate" in WEAVIATE_API_KEY:
        missing.append("WEAVIATE_API_KEY")
    if not OPENAI_API_KEY or "your-openai" in OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if missing:
        print(f"\n❌ Missing values in .env: {', '.join(missing)}")
        sys.exit(1)

    # ── Handle reset ──
    if args.reset:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            print("🗑️  Progress log cleared — will re-tag all chunks.")
        else:
            print("ℹ️  No progress log found — starting fresh anyway.")

    # ── Load progress log ──
    tagged_uuids = load_log()
    if tagged_uuids:
        print(f"📋 Resuming — {len(tagged_uuids):,} chunks already tagged, skipping those.\n")

    # ── Connect to Weaviate ──
    print(f"🔌 Connecting to Weaviate Cloud...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
        )
        collection = client.collections.get(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Could not connect to Weaviate: {e}")
        sys.exit(1)

    print("✅ Connected!")

    # ── Add new Phase 2 properties if they don't exist yet ──
    # This is safe to run even if the properties already exist — it checks first.
    print("🔧 Checking schema for new Phase 2 fields...")
    try:
        from weaviate.classes.config import Property, DataType
        existing_props = {p.name for p in collection.config.get().properties}

        new_props = [
            ("quotes_present",      DataType.BOOL, "[Phase 2] True if chunk contains a direct quote from a fighter, trainer, or eyewitness"),
            ("controversy_present", DataType.BOOL, "[Phase 2] True if chunk describes a disputed result, controversial decision, or scandal"),
        ]
        for prop_name, dtype, desc in new_props:
            if prop_name not in existing_props:
                collection.config.add_property(
                    Property(name=prop_name, data_type=dtype, description=desc, skip_vectorization=True)
                )
                print(f"   ✅ Added new property: {prop_name}")
            else:
                print(f"   ✓  Already exists:     {prop_name}")
    except Exception as e:
        print(f"   ⚠️  Schema check failed: {e}")
        print("   Continuing anyway — properties may already exist.\n")

    print()

    # ── Count total chunks ──
    try:
        total = collection.aggregate.over_all(total_count=True).total_count
        print(f"📦 Archive contains {total:,} total chunks.")
        remaining = total - len(tagged_uuids)
        print(f"🏷️  Chunks to tag this run: {remaining:,}\n")
    except Exception:
        total = 0
        print("📦 Could not get total count — proceeding anyway.\n")

    # ── Connect OpenAI ──
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # ── Iterate and tag ──
    tagged_this_run  = 0
    failed_this_run  = 0
    skipped_this_run = 0

    print("=" * 60)
    print("  Starting Phase 2 tagging...")
    print("  You can stop at any time with Ctrl+C — progress is saved.")
    print("=" * 60 + "\n")

    try:
        for obj in collection.iterator(
            return_properties=["content", "source_file", "title"]
        ):
            uuid = str(obj.uuid)

            # Skip already tagged
            if uuid in tagged_uuids:
                skipped_this_run += 1
                continue

            content = obj.properties.get("content", "")
            source  = obj.properties.get("source_file", "unknown")

            if not content.strip():
                # Empty chunk — write blank tags and mark done
                collection.data.update(
                    uuid=uuid,
                    properties={
                        "weight_class_focus":         "unknown",
                        "fighters_mentioned":         "none",
                        "venues_mentioned":           "none",
                        "contains_training_methods":  False,
                        "contains_fight_account":     False,
                        "contains_biographical_info": False,
                        "has_statistics":             False,
                        "quotes_present":             False,
                        "controversy_present":        False,
                    }
                )
                tagged_uuids.add(uuid)
                save_log(tagged_uuids)
                continue

            # Tag the chunk
            tags_raw = tag_chunk(openai_client, content[:2000])  # cap at 2000 chars

            if tags_raw is None:
                failed_this_run += 1
                # Don't add to log — will retry on next run
                time.sleep(1.0)
                continue

            tags = validate_tags(tags_raw)

            # Write tags back to Weaviate
            try:
                collection.data.update(uuid=uuid, properties=tags)
                tagged_uuids.add(uuid)
                tagged_this_run += 1

                # Save progress every 10 chunks
                if tagged_this_run % 10 == 0:
                    save_log(tagged_uuids)

                # Progress output every 50 chunks
                if tagged_this_run % 50 == 0:
                    total_done = len(tagged_uuids)
                    pct = (total_done / total * 100) if total else 0
                    print(f"   ✅ {total_done:,} / {total:,} tagged ({pct:.1f}%)")

                # Small print for every chunk so you can see it's working
                fighters = tags["fighters_mentioned"]
                f_str = fighters[:60] if fighters != "none" else "—"
                print(f"   [{tagged_this_run:>4}] {source[:45]:<45}  fighters: {f_str}")

            except Exception as e:
                print(f"   ⚠️  Failed to update chunk in Weaviate: {e}")
                failed_this_run += 1

            time.sleep(RATE_LIMIT_PAUSE)

    except KeyboardInterrupt:
        print("\n\n⏸️  Stopped by user.")

    finally:
        # Always save progress on exit
        save_log(tagged_uuids)
        client.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  PHASE 2 TAGGING SUMMARY")
    print("=" * 60)
    print(f"  Tagged this run:     {tagged_this_run:,}")
    print(f"  Failed this run:     {failed_this_run:,}")
    print(f"  Skipped (done prev): {skipped_this_run:,}")
    print(f"  Total tagged so far: {len(tagged_uuids):,}")

    if failed_this_run > 0:
        print(f"\n  ⚠️  {failed_this_run} chunks failed — run the script again to retry them.")

    if len(tagged_uuids) >= total and total > 0:
        print("\n  🎉 All chunks tagged! Phase 2 complete.")
        print("  Your archive now supports filtered searches by fighter,")
        print("  venue, weight class, training methods, fight accounts, and stats.")
    else:
        remaining = total - len(tagged_uuids)
        print(f"\n  👉 {remaining:,} chunks still to go — run the script again to continue.")

    print()


if __name__ == "__main__":
    main()
