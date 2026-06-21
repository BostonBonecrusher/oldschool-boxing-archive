"""
app.py
------
Oldschool Gladiators Boxing Archive — Web Application

Run locally:
    python app.py
    Then open http://localhost:5000 in your browser.

Deploy to Render:
    1. Push this project folder to GitHub
    2. Connect Render to that repo
    3. Set environment variables in Render dashboard (same as your .env)
    4. Deploy — Render runs: gunicorn app:app
"""

import os
import re
import json
import random
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import weaviate
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter
from openai import OpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
APP_PASSWORD     = os.getenv("APP_PASSWORD", "gladiators")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-before-deploying")

# ── Topics ────────────────────────────────────────────────────────────────────
# Two archives, one app. Add a new entry here (+ a system prompt + keyword
# lists below) to plug in a third topic later without touching the routes.

DEFAULT_TOPIC = "boxing"

TOPICS = {
    "boxing": {
        "collection_name": "BoxingChunk",
        "label":    "Boxing History",
        "tagline":  "Sourced answers from primary boxing texts — no guessing, no modern bias.",
        "placeholder": "e.g. How did Harry Greb train outside of fighting?",
        "examples": [
            "Sam Langford stories",
            "How did Jack Dempsey develop his punch?",
            "Greb's fight record",
            "Why did boxing decline after the 1950s?",
            "Henry Armstrong's training methods",
            "Jack Johnson controversies",
        ],
        "subject_field":   "subject_fighter",
        "mentioned_field": "fighters_mentioned",
        "has_phase2": True,
    },
    "warfare": {
        "collection_name": "HistoricalWarfareChunk",
        "label":    "Historical Warfare",
        "tagline":  "Sourced answers from primary military texts — old-world training, tactics, and command.",
        "placeholder": "e.g. How did Roman legionaries train and condition for campaign?",
        "examples": [
            "Roman legion daily training",
            "What did soldiers eat on campaign?",
            "Hannibal's tactics at Cannae",
            "How was discipline enforced in the Roman army?",
            "Byzantine cavalry tactics",
            "Fall of Constantinople 1453",
        ],
        "subject_field":   "subject_commander",
        "mentioned_field": "commanders_mentioned",
        "has_phase2": False,   # commanders_mentioned/battles_mentioned/etc. exist in the
                                # schema but are still blank — no Phase 2 tagging pass has
                                # been run on this collection yet. Intent filtering for this
                                # topic uses the populated Phase-1 `discipline` field instead.
    },
}


def _topic_key(raw):
    """Validate an incoming topic string, falling back to the default."""
    key = (raw or "").strip().lower()
    return key if key in TOPICS else DEFAULT_TOPIC


# ── Story Bank (Story Ideas / Browse by name / Filter by era) ─────────────────
# story_hooks.json is written by mine_stories.py every time it exports a
# collection. It's a curated subset (only the best-scoring stories) keyed by
# topic, e.g. {"boxing": [...], "warfare": [...]}. Loaded once at startup --
# if Jon mines more stories later, a redeploy picks up the refreshed file.

STORY_HOOKS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "story_hooks.json")
story_hooks = {}       # topic key -> list of {hook, category, people, wow, era}
story_people = {}      # topic key -> sorted list of distinct people names
story_eras = {}        # topic key -> sorted list of distinct era codes

# era codes are free-text in the schema but follow a per-topic convention
# (see 01_create_schema.py / 01_create_schema_warfare.py). Human-friendly
# labels for the "Filter by era" dropdown; anything unrecognized just gets
# title-cased rather than failing.
ERA_LABELS = {
    "boxing": {
        "bare_knuckle_era": "Bare-Knuckle Era (pre-1900)",
        "golden_age":       "Golden Age (1900-1950)",
        "midcentury":       "Midcentury (1950-1980)",
        "modern":           "Modern (1980+)",
        "unknown":          "Unknown Era",
    },
    "warfare": {
        "ancient":               "Ancient (pre-500 CE)",
        "byzantine":             "Byzantine (500-1453 CE)",
        "ottoman":               "Ottoman (1300-1922 CE)",
        "early_modern_conquest": "Early Modern Conquest (1492-1700 CE)",
        "modern":                "Modern (1700 CE+)",
        "multi_era":             "Multiple Eras",
        "unknown":               "Unknown Era",
    },
}


def _era_label(topic, code):
    if not code:
        return "Unknown Era"
    return ERA_LABELS.get(topic, {}).get(code, code.replace("_", " ").title())


def load_story_hooks():
    global story_hooks, story_people, story_eras
    data = {}
    if os.path.exists(STORY_HOOKS_FILE):
        try:
            with open(STORY_HOOKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    story_hooks = data
    story_people = {}
    story_eras = {}
    for topic, items in data.items():
        names = set()
        eras = set()
        for item in items:
            for person in item.get("people", []):
                if person:
                    names.add(person)
            if item.get("era"):
                eras.add(item["era"])
        story_people[topic] = sorted(names)
        story_eras[topic] = sorted(eras)


load_story_hooks()


TOP_K_DEFAULT = 10
TOP_K_RECORDS = 15

NARRATIVE_KEYWORDS = {
    "boxing": [
        'typical day', 'daily routine', 'what was it like', 'describe', 'camp',
        'training camp', 'preparation', 'prepare', 'how did he train', 'style',
        'fighting style', 'personality', 'what kind of', 'tell me about',
        'atmosphere', 'lifestyle', 'roadwork', 'sparring', 'gym', 'diet',
        'conditioning', 'routine', 'approach', 'method', 'technique', 'character',
        'reputation', 'known for', 'famous for', 'what made', 'how was he',
        'what were his', 'how did', 'how were', 'how would', 'what did they',
        'start to finish', 'walk me through', 'explain how', 'what was the',
        'what is the', 'tell us about', 'give me', 'overview', 'breakdown',
        'in detail', 'background on', 'history of',
    ],
    "warfare": [
        'typical day', 'daily routine', 'what was it like', 'describe', 'camp',
        'life as a soldier', 'campaign life', 'how did they train', 'training',
        'discipline', 'diet', 'rations', 'conditioning', 'drill', 'march',
        'formation', 'siege', 'command', 'leadership', 'style', 'approach',
        'method', 'technique', 'character', 'reputation', 'known for',
        'famous for', 'what made', 'how was he', 'what were his', 'how did',
        'how were', 'how would', 'what did they', 'start to finish',
        'walk me through', 'explain how', 'what was the', 'what is the',
        'tell us about', 'tell me about', 'give me', 'overview', 'breakdown',
        'in detail', 'background on', 'history of', 'tactics', 'strategy',
    ],
}

RECORDS_KEYWORDS = {
    "boxing": [
        'record', 'fight', 'won', 'lost', 'draw', 'knockout', 'ko', 'total fights',
        'how many', 'wins', 'losses', 'undefeated', 'career', 'times', 'weight',
        'pounds', 'lbs', 'champion', 'title', 'reign', 'held', 'defended',
    ],
    "warfare": [
        'how many', 'troops', 'soldiers', 'men', 'casualties', 'losses',
        'killed', 'wounded', 'strength', 'army size', 'numbers', 'total force',
        'duration', 'how long', 'campaign length', 'legions', 'ships',
    ],
}
TEMPORAL_KEYWORDS = {
    "boxing": [
        'early career', 'late career', 'end of career', 'beginning', 'later',
        'prime', 'young', 'old', 'retire', 'final', 'last fight', 'first fight',
        'at the time', 'during', 'before', 'after', 'by the time',
    ],
    "warfare": [
        'early reign', 'late empire', 'decline of', 'rise of', 'fall of',
        'reign of', 'beginning', 'later', 'final years', 'before the battle',
        'after the battle', 'at the time', 'during', 'before', 'after',
        'by the time', 'campaign',
    ],
}

# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ── Global clients ────────────────────────────────────────────────────────────

weaviate_client = None
collections     = {}   # topic key -> weaviate Collection handle
openai_client   = None
archive_counts  = {}   # topic key -> total chunk count


def init_clients():
    global weaviate_client, collections, openai_client, archive_counts
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=120, insert=180)
        ),
    )
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    for topic, cfg in TOPICS.items():
        coll = weaviate_client.collections.get(cfg["collection_name"])
        collections[topic] = coll
        try:
            archive_counts[topic] = coll.aggregate.over_all(total_count=True).total_count
        except Exception:
            archive_counts[topic] = 0


def get_collection(topic):
    return collections.get(topic) or collections.get(DEFAULT_TOPIC)


# ── Auth ──────────────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("password") == APP_PASSWORD:
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Incorrect password. Try again."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    return render_template(
        "index.html",
        default_topic=DEFAULT_TOPIC,
        topics=TOPICS,
        chunk_counts={t: f"{c:,}" for t, c in archive_counts.items()},
    )


@app.route("/stitch", methods=["POST"])
@login_required
def stitch():
    """
    Given a source_file and chunk_index, fetch the surrounding chunks
    and return them stitched together as a continuous passage.
    """
    data        = request.get_json()
    topic       = _topic_key(data.get("topic"))
    source_file = (data.get("source_file") or "").strip()
    chunk_index = int(data.get("chunk_index", 0))
    radius      = int(data.get("radius", 2))   # chunks before + after to fetch

    if not source_file:
        return jsonify({"error": "source_file required"}), 400

    try:
        start = max(0, chunk_index - radius)
        end   = chunk_index + radius
        collection = get_collection(topic)

        results = collection.query.fetch_objects(
            filters=Filter.all_of([
                Filter.by_property("source_file").equal(source_file),
                Filter.by_property("chunk_index").greater_or_equal(start),
                Filter.by_property("chunk_index").less_or_equal(end),
            ]),
            limit=radius * 2 + 3,
            return_properties=["content", "chunk_index"],
        )

        # Sort by chunk_index and stitch
        chunks = sorted(results.objects, key=lambda o: o.properties.get("chunk_index", 0))
        stitched = "\n\n".join(o.properties.get("content", "") for o in chunks)

        return jsonify({
            "stitched":    stitched,
            "chunk_start": chunks[0].properties.get("chunk_index", start) if chunks else start,
            "chunk_end":   chunks[-1].properties.get("chunk_index", end)  if chunks else end,
            "count":       len(chunks),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/story-people", methods=["GET"])
@login_required
def story_people_route():
    """Distinct people names available in this topic's curated story bank,
    for the 'Browse by name' picker."""
    topic = _topic_key(request.args.get("topic"))
    return jsonify({"people": story_people.get(topic, [])})


@app.route("/story-eras", methods=["GET"])
@login_required
def story_eras_route():
    """Distinct eras available in this topic's curated story bank, for the
    'Filter by era' picker. Returns {value, label} pairs since era codes
    (e.g. 'bare_knuckle_era') aren't display-friendly on their own."""
    topic = _topic_key(request.args.get("topic"))
    codes = story_eras.get(topic, [])
    return jsonify({"eras": [{"value": c, "label": _era_label(topic, c)} for c in codes]})


@app.route("/random-stories", methods=["GET"])
@login_required
def random_stories():
    """
    Story Ideas / Browse by name / Filter by era. With no filters, returns
    up to 5 random curated hooks (the 'surprise me' button). With ?person=
    and/or ?era= (combinable), returns up to 25 matching stories per page,
    sorted by wow -- someone who picked a specific filter wants the best
    matches, not a random sample. Pass ?offset=25, ?offset=50, etc. to page
    through the rest; the response's "has_more" flag tells the caller
    whether another page exists.
    """
    PAGE_SIZE = 25
    topic  = _topic_key(request.args.get("topic"))
    person = (request.args.get("person") or "").strip()
    era    = (request.args.get("era") or "").strip()
    offset = max(0, request.args.get("offset", 0, type=int) or 0)
    pool   = story_hooks.get(topic, [])

    if not pool:
        return jsonify({
            "stories": [],
            "message": f"No curated stories yet for {TOPICS[topic]['label']} -- "
                       f"check back once more of the archive has been mined.",
        })

    if person or era:
        matches = pool
        if person:
            matches = [s for s in matches if person in s.get("people", [])]
        if era:
            matches = [s for s in matches if s.get("era") == era]
        matches = sorted(matches, key=lambda s: s.get("wow", 0), reverse=True)
        total = len(matches)
        page  = matches[offset:offset + PAGE_SIZE]
        if not page:
            who  = f" mentioning {person}" if person else ""
            when = f" from the {_era_label(topic, era)}" if era else ""
            msg  = "No more stories." if offset else f"No curated stories yet{who}{when}."
            return jsonify({"stories": [], "total": total, "has_more": False, "message": msg})
        return jsonify({
            "stories": page,
            "total": total,
            "has_more": offset + PAGE_SIZE < total,
        })

    n = min(5, len(pool))
    return jsonify({"stories": random.sample(pool, n)})


@app.route("/search", methods=["POST"])
@login_required
def search():
    data = request.get_json()
    topic    = _topic_key(data.get("topic"))
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        is_records  = any(k in question.lower() for k in RECORDS_KEYWORDS[topic])
        is_temporal = any(k in question.lower() for k in TEMPORAL_KEYWORDS[topic])

        # Named-entity / intent detection (for response metadata + filtering)
        names  = _extract_proper_names(question, topic)
        intent = _detect_content_intent(question, topic)

        # Retrieve chunks
        if is_records or is_temporal:
            chunks = _search_expanded(question, topic)
        else:
            chunks = _search_phase2(question, topic)

        if not chunks:
            return jsonify({"error": "No relevant passages found in the archive for that question."})

        context = _build_context(chunks)
        answer  = _ask_ai(question, context, topic, is_records, is_temporal)

        subject_field = TOPICS[topic]["subject_field"]

        # Serialize sources for the frontend
        sources = []
        for i, chunk in enumerate(chunks, 1):
            props   = chunk.properties
            content = props.get("content", "")
            sources.append({
                "index":   i,
                "title":   props.get("title") or props.get("source_file", "Unknown"),
                "author":  props.get("author", ""),
                "year":    props.get("year_published", 0) or 0,
                "primary": bool(props.get("is_primary_source", False)),
                "doc_type":props.get("document_type", ""),
                "era":     props.get("era", ""),
                "geo":     props.get("geographic_focus", ""),
                "subject": props.get(subject_field, ""),
                "ocr":     bool(props.get("ocr_used", False)),
                "score":       round(chunk.metadata.score, 3) if chunk.metadata else 0,
                "content":     content,
                "preview":     content[:380] + ("…" if len(content) > 380 else ""),
                "source_file": props.get("source_file", ""),
                "chunk_index": props.get("chunk_index", 0),
            })

        return jsonify({
            "answer":  answer,
            "sources": sources,
            "flags": {
                "records":       is_records,
                "temporal":      is_temporal,
                "names":         names,
                "intent":        sorted(intent),
            },
        })

    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)}"}), 500


# ── Search helpers ────────────────────────────────────────────────────────────

# ── Phase 2 helpers ───────────────────────────────────────────────────────────

_NAME_STOPWORDS_BASE = {
    'The', 'What', 'How', 'Who', 'When', 'Where', 'Why', 'Did', 'Was', 'Is',
    'Tell', 'Can', 'Could', 'Would', 'Should', 'In', 'At', 'On', 'By', 'Of',
    'His', 'Her', 'Their', 'He', 'She', 'They', 'It', 'And', 'Or', 'But',
    'For', 'With', 'About', 'From', 'Between', 'During', 'After', 'Before',
    'Old', 'New', 'First', 'Last', 'Early', 'Late', 'American', 'British',
    'World', 'Title', 'Era', 'History', 'Prime', 'Final', 'Known', 'Famous',
    'Great', 'Years', 'Days', 'Life', 'Lives', 'York', 'London', 'Book', 'Vol',
}

_NAME_STOPWORDS = {
    "boxing": _NAME_STOPWORDS_BASE | {
        'Championship', 'Fight', 'Fights', 'Bout', 'Round', 'Boxing',
        'Career', 'Nights', 'Night', 'Stateside', 'Scraps', 'Scrapes',
        'Scuffles', 'Tales', 'Stories', 'Story', 'Ring', 'Corner', 'Science',
        'Art', 'Guide', 'Methods', 'Secrets', 'Memoirs', 'Chicago', 'Gazette',
        'Journal', 'Record', 'Press', 'Tribune', 'Times', 'Arc', 'Gladiators',
        'Angels', 'Sweet', 'Craft',
    },
    "warfare": _NAME_STOPWORDS_BASE | {
        'Rome', 'Roman', 'Romans', 'Empire', 'Republic', 'War', 'Wars',
        'Battle', 'Battles', 'Legion', 'Legions', 'Army', 'Ancient',
        'Byzantine', 'Ottoman', 'Greek', 'Greece', 'Constantinople',
        'Egypt', 'Egyptians', 'Persia', 'Persians', 'Arab', 'Arabs',
        'Conquest', 'Siege', 'Campaign', 'Decline', 'Fall', 'Progress',
        'Termination', 'Tactics', 'Strategy', 'History', 'Volume', 'Vols',
    },
}

_SOURCE_TRIGGERS = [
    'in ', 'from ', 'check ', 'according to ', 'it says in ', 'states in ',
    'mentioned in ', 'look in ', 'what does ', 'that book', 'the book',
]


def _extract_source_reference(question):
    q_lower = question.lower()
    found = []
    for trigger in _SOURCE_TRIGGERS:
        idx = q_lower.find(trigger)
        if idx == -1:
            continue
        fragment = question[idx + len(trigger):idx + len(trigger) + 80]
        words = re.findall(r'[A-Za-z]{3,}', fragment)
        for w in words:
            wl = w.lower()
            if (w[0].isupper() or wl in {
                'stateside', 'scraps', 'scuffles', 'scrapes', 'nights',
                'fight', 'nights', 'york', 'london', 'ring', 'corner',
                'gladiators', 'angels', 'sweet', 'science', 'arc',
            }) and wl not in {'the', 'and', 'for', 'with', 'that', 'this',
                               'from', 'about', 'what', 'does', 'book'}:
                found.append(wl)
            if len(found) >= 4:
                break
        if found:
            break
    return found


def _build_source_filter(title_words):
    if not title_words:
        return None
    conditions = []
    for word in title_words[:3]:
        if len(word) > 3:
            conditions.append(Filter.by_property("title").like(f"*{word}*"))
            conditions.append(Filter.by_property("source_file").like(f"*{word}*"))
    if not conditions:
        return None
    return Filter.any_of(conditions) if len(conditions) > 1 else conditions[0]

_INTENT_MAP_BOXING = {
    "fight_account":   ["round", "knocked", "knockdown", "bout", " vs ", " versus ",
                        "beat", "defeated", "won the fight", "lost the fight",
                        "how did the fight", "fight go", "what happened in the",
                        "describe the fight", "fell in", "down in round"],
    "biographical":    ["life", "born", "grew up", "childhood", "personal history",
                        "background", "early life", "family", "where was",
                        "tell me about", "who was", "biography"],
    "training":        ["train", "conditioning", "workout", "method", "technique",
                        "sparring", "roadwork", "gym", "drill", "exercise",
                        "preparation", "daily routine", "how did he prepare"],
    "quotes":          ["said", "quote", "in his own words", "what did",
                        "stated", "wrote", "described", "according to him",
                        "his words", "commented", "remarked"],
    "controversy":     ["disputed", "controversial", "robbery", "fixed", "rigged",
                        "scandal", "cheat", "bias", "corrupt", "complaint",
                        "protest", "bad decision", "stolen", "was it fixed"],
    "statistics":      ["record", "wins", "losses", "knockouts", "how many fights",
                        "total fights", "fight record", "career stats"],
}

# HistoricalWarfareChunk has no Phase 2 AI tags yet (commanders_mentioned,
# contains_battle_account, etc. are defined in the schema but still blank —
# no tagging pass has been run on this collection). Use the populated
# Phase-1 `discipline` field as a coarser stand-in for intent filtering.
_DISCIPLINE_INTENT_MAP = {
    "tactics_and_strategy":           ["tactics", "strategy", "formation", "maneuver",
                                        "manoeuvre", "outflank", "ambush"],
    "siege_warfare":                  ["siege", "fortification", "walls", "breach",
                                        "besiege"],
    "leadership_and_command":         ["command", "leadership", "general", "commander",
                                        "led his", "in charge"],
    "administration_and_discipline":  ["discipline", "training", "drill", "punishment",
                                        "conditioning", "regulation", "rations", "diet"],
    "military_medicine":              ["wound", "medicine", "injury", "surgeon", "healing"],
    "campaign_history":                ["campaign", "march", "invasion", "expedition"],
    "cavalry":                         ["cavalry", "horse", "mounted", "horsemen"],
    "religion_and_culture":            ["religion", "ritual", "culture", "belief", "worship"],
    "diplomacy_and_policy":            ["diplomacy", "treaty", "policy", "alliance", "envoy"],
}

_RETURN_PROPS = {
    "boxing": [
        "content", "source_file", "title", "author",
        "year_published", "document_type", "era", "rules_era",
        "discipline", "is_primary_source", "subject_fighter",
        "geographic_focus", "page_number", "chunk_index", "ocr_used",
        "fighters_mentioned", "weight_class_focus",
    ],
    "warfare": [
        "content", "source_file", "title", "author",
        "year_published", "document_type", "era", "time_period",
        "discipline", "is_primary_source", "subject_commander",
        "civilization_or_nation", "conflict_focus", "geographic_focus",
        "page_number", "chunk_index", "ocr_used", "commanders_mentioned",
    ],
}


def _extract_proper_names(question, topic):
    tokens = re.findall(r"[A-Z][a-z]{2,}", question)
    stop = _NAME_STOPWORDS.get(topic, _NAME_STOPWORDS_BASE)
    return [t for t in tokens if t not in stop]


def _detect_content_intent(question, topic):
    q = question.lower()
    detected = set()
    intent_map = _INTENT_MAP_BOXING if TOPICS[topic]["has_phase2"] else _DISCIPLINE_INTENT_MAP
    for intent, keywords in intent_map.items():
        if any(kw in q for kw in keywords):
            detected.add(intent)
    return detected


def _build_subject_filter(names, topic):
    """Filter on whichever field is actually populated for this topic:
    boxing has Phase-2 `fighters_mentioned`; warfare only has Phase-1
    `subject_commander` populated so far."""
    if not names:
        return None
    field = TOPICS[topic]["mentioned_field"] if TOPICS[topic]["has_phase2"] else TOPICS[topic]["subject_field"]
    conditions = [
        Filter.by_property(field).like(f"*{name}*")
        for name in names if len(name) > 2
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


def _build_intent_filter(intent, topic):
    if TOPICS[topic]["has_phase2"]:
        field_map = {
            "fight_account":   ("contains_fight_account",     True),
            "biographical":    ("contains_biographical_info",  True),
            "training":        ("contains_training_methods",   True),
            "quotes":          ("quotes_present",              True),
            "controversy":     ("controversy_present",         True),
            "statistics":      ("has_statistics",              True),
        }
        conditions = [
            Filter.by_property(field).equal(val)
            for key, (field, val) in field_map.items()
            if key in intent
        ]
    else:
        # warfare: intent values ARE discipline values directly
        conditions = [
            Filter.by_property("discipline").equal(discipline_value)
            for discipline_value in intent
        ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


def _run_hybrid(query, top_k, topic, filters=None):
    kwargs = dict(
        query=query,
        limit=top_k,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
        return_properties=_RETURN_PROPS[topic],
    )
    if filters is not None:
        kwargs["filters"] = filters
    return get_collection(topic).query.hybrid(**kwargs).objects


def _merge_unique(lists):
    seen, merged = set(), []
    for lst in lists:
        for chunk in lst:
            key = chunk.properties.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged


def _search_archive(query, topic, top_k=None):
    """Legacy direct search — used internally and for fallback."""
    is_records = any(k in query.lower() for k in RECORDS_KEYWORDS[topic])
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records else TOP_K_DEFAULT
    return _run_hybrid(query, top_k, topic)


def _search_phase2(query, topic, top_k=None):
    """
    Streamlined search — single hybrid query with optional subject/source filter combined.
    Avoids multiple sequential gRPC calls that cause Deadline Exceeded on large collections.
    """
    is_records = any(k in query.lower() for k in RECORDS_KEYWORDS[topic])
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records else TOP_K_DEFAULT

    names       = _extract_proper_names(query, topic)
    source_refs = _extract_source_reference(query)

    # Try one filtered search first (subject or source), then baseline
    combined_filter = None
    source_filter  = _build_source_filter(source_refs)
    subject_filter = _build_subject_filter(names, topic)

    if source_filter is not None:
        combined_filter = source_filter
    elif subject_filter is not None:
        combined_filter = subject_filter

    result_sets = []
    if combined_filter is not None:
        result_sets.append(_run_hybrid(query, top_k, topic, filters=combined_filter))

    # Always run baseline
    result_sets.append(_run_hybrid(query, top_k, topic))

    merged = _merge_unique(result_sets)
    return merged[:top_k + 8]


def _search_expanded(query, topic):
    # Keep it to one extra pass max
    results1 = _search_phase2(query, topic)
    words = query.lower().split()
    secondary = " ".join(
        [w for w in query.split() if w[0].isupper()] +
        [w for w in words if w in RECORDS_KEYWORDS[topic]]
    )
    if not secondary.strip():
        return results1
    results2 = _run_hybrid(secondary, top_k=8, topic=topic)
    return _merge_unique([results1, results2])


def _build_context(chunks):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        props   = chunk.properties
        title   = props.get("title") or props.get("source_file", "Unknown")
        author  = props.get("author", "")
        year    = props.get("year_published", 0)
        primary = props.get("is_primary_source", False)
        chunk_i = props.get("chunk_index", 0)
        content = props.get("content", "")
        citation = title
        if author:  citation += f" by {author}"
        if year:    citation += f" ({year})"
        if primary: citation += " [PRIMARY SOURCE]"
        citation += f" [passage {chunk_i}]"
        parts.append(f"[Source {i}: {citation}]\n{content}")
    return "\n\n---\n\n".join(parts)


_SYSTEM_PROMPT_BOXING = """You are a boxing historian and research assistant for a YouTube channel \
and paid archive called Oldschool Gladiators, which specializes in pre-1950s boxing history. \
Your answers are used directly in published content and a paid research product. \
Absolute factual accuracy and source integrity are non-negotiable. \
One wrong fact destroys the credibility of everything.

═══════════════════════════════════════════════════════
FOUNDATIONAL PHILOSOPHY — READ THIS BEFORE ALL RULES
═══════════════════════════════════════════════════════

This archive exists to document and honour the fighters and methods of the pre-television \
era of boxing — roughly pre-1950s. You must approach all questions with the following \
understanding deeply embedded in your reasoning:

1. OLDER FIGHTERS ARE NOT INFERIOR BY DEFAULT.
The assumption that modern fighters are superior is a modern media construct, not a \
historical fact. Pre-1950s fighters fought far more frequently, over more rounds, \
against deeper competition, with less recovery time, less protective equipment, and \
without the benefit of modern sports science. These conditions produced fighters of \
extraordinary durability, conditioning, and craft. Do not frame pre-1950s fighters \
as primitive, limited, or products of a weaker era. They are the standard.

2. UNDERSTAND HOW BOXING CHANGED AFTER TELEVISION.
When television took over boxing in the late 1940s and 1950s, the sport fundamentally \
changed — not necessarily for the better. Fight frequency dropped dramatically. \
Promotional machinery, marketing, and narrative replaced pure fighting merit as the \
primary driver of a fighter's reputation. Modern "legends" are often partly products \
of media construction. Pre-TV fighters were judged almost exclusively on results \
against other fighters. Keep this context in mind when comparing eras.

3. PRIMARY SOURCES FROM THE ERA ARE THE GOLD STANDARD.
A book written in 1921 about a fighter's methods carries more authenticity than a \
modern biography written 100 years later. When a pre-1950s primary source describes \
a training method or a fighter's attributes, treat it as the most credible account \
available — not as a quaint historical curiosity.

4. DO NOT IMPORT MODERN BIAS INTO HISTORICAL ANALYSIS.
Do not use phrases like "by modern standards" or "in today's terms" in ways that \
imply modern boxing is the benchmark. If anything, the reverse framing is more \
historically accurate — modern boxing should be measured against the pre-TV era, \
not the other way around.

5. MARKETING VS MERIT.
When comparing fighters across eras or discussing reputations, always consider \
the role of marketing, television exposure, and promotional machinery in shaping \
modern perceptions. A fighter with 300 fights and no promotional machine behind him \
may be historically more significant than a modern fighter with 30 fights and a \
billion-dollar promotional contract.

6. UNDERSTAND BOXING'S STRUCTURAL DECLINE — THE ARC OF BOXING FRAMEWORK.
The archive includes "The Arc of Boxing" which provides the core analytical framework \
for understanding how and why boxing declined. Apply this framework when evaluating \
fighters, eras, records, and reputations:

  THE DECLINE WAS STRUCTURAL, NOT SUDDEN.
  Boxing did not decline because fans disappeared. It declined because its \
infrastructure was hollowed out after World War II — fewer young men chose boxing \
as a career, neighbourhood gyms and arenas closed, suburbanisation pulled the \
sport's core audience away from urban fight clubs, and television changed how \
fans consumed the sport. The result was a steadily shrinking talent pipeline.

  THE APPRENTICESHIP SYSTEM WAS LOST.
  Pre-decline boxing developed fighters gradually through a dense network of \
experienced trainers, managers, and frequent competition. Fighters built their \
craft over hundreds of bouts before facing elite opposition. That system produced \
technically superior fighters because it prioritised development over spectacle. \
When that system collapsed, so did the depth and quality of the sport.

  PROTECTING RECORDS REPLACED HARD COMPETITION.
  Modern boxing is structurally incentivised to protect records and produce \
marketable spectacles rather than to build fighters through hard competition. \
This directly weakens the sport. A pre-1950s fighter with 300 fights built \
genuine ring intelligence through adversity. A modern fighter with a carefully \
managed 30-fight record has been shielded from that development process.

  APPLY THIS FRAMEWORK WHEN COMPARING ERAS.
  When a question involves comparing fighters or eras, consider: what was the \
infrastructure around that fighter? How was he developed? How frequently did he \
compete? What was the depth of opposition available to him? A fighter produced \
by the old apprenticeship system against deep opposition is not directly \
comparable to a modern fighter produced by a promotional machine against \
a managed schedule. Note these structural differences explicitly.

7. SIZE AND WEIGHT ARE ONE FACTOR AMONG MANY — NOT THE LEADING FACTOR.
Boxing history repeatedly demonstrates that size and weight advantages are frequently \
overcome and are not the primary determinant of fighting ability or career merit. \
Sam Langford — one of the most skilled fighters in history — routinely defeated men \
significantly larger than himself. Joe Louis destroyed bigger men with efficiency \
and precision. Rocky Marciano was undersized for a heavyweight yet went undefeated. \
Henry Armstrong held three titles simultaneously at weights far below the men he fought.

When analysing or comparing fighters, treat size and weight as ONE contextual factor \
among many — equal in weight (no pun intended) to: fight volume, quality of opposition, \
era of competition, conditioning, ring craft, durability, and adaptability. \
Never lead with size as an explanation for a result or a career assessment. \
Never use size disparity to diminish a smaller fighter's achievement or to \
excuse a larger fighter's loss.

When a source notes a size or weight discrepancy, present it as context — not as \
a determining factor. The record of what actually happened in the ring is always \
more important than a pre-fight physical comparison.

Specifically: do not use modern sports science assumptions about weight classes \
and size advantages to retroactively judge pre-1950s fighters. The weight class \
system was far less rigid, fighters routinely competed across multiple weight classes, \
and the sport's history shows that craft, conditioning, and experience routinely \
trumped size.

═══════════════════════════════════════════
CORE RULES — NEVER VIOLATE ANY OF THESE
═══════════════════════════════════════════

RULE 1 — SOURCES ONLY.
Use ONLY the provided source passages. Never add facts, dates, names, records, or claims \
from your own training data. If it is not explicitly written in the sources provided, \
do not say it. Not even if you are certain it is true.

RULE 2 — IF SOURCES ARE PROVIDED, YOU MUST ANSWER.
This is absolute. If source passages have been retrieved and provided to you, \
you are required to attempt a full answer using them. \
"The archive does not contain sufficient information" is ONLY permitted when \
the retrieved passages contain zero relevant content — meaning they are \
entirely off-topic and share no connection to the question at all. \
It is NEVER acceptable to refuse when the sources contain partial, \
tangential, or incomplete but relevant information.

  FACTUAL QUESTIONS (exact records, dates, specific fight results):
  Cite the specific figures from the sources. If the exact fact isn't present, \
  say what IS present and note the gap. Do not fabricate. Do not refuse.

  ALL OTHER QUESTIONS — SYNTHESIS IS REQUIRED:
  No single passage needs to fully answer the question. \
  Assemble the picture from every relevant detail across all provided sources \
  and cite each one as you go. This is historical research, not a keyword lookup. \
  If a question asks about Roman legion marching and the sources contain passages \
  about drill, endurance, discipline, or conditioning — use them. \
  If the picture is partial, open with: \
  "The archive offers a partial but vivid account of..." and present everything \
  the sources contain. A partial answer built from real sources is always \
  more valuable than a refusal.

RULE 3 — CITE EVERY SINGLE CLAIM.
Every statement of fact must be tied to a source. Format:
"According to [Source Title, Year]..." or "As recorded in [Source Title, Year]..."
Never make an unsourced statement.

RULE 4 — FIGHT RECORDS REQUIRE SPECIAL HANDLING.
Records (wins, losses, draws, KOs, total fights, weight, title reigns) are the most \
verifiable and most scrutinized data in boxing history. Handle them as follows:
  a) NEVER combine, average, or blend record figures from different sources.
  b) If multiple sources give different figures, list EACH source's figure separately.
  c) If a question asks about a specific career period and the sources contain figures \
from different periods, make that distinction explicit.
  d) If only one source covers the specific statistic asked about, present that figure \
and note it comes from one source only.

RULE 5 — PRIMARY SOURCES GET PRIORITY.
Sources marked [PRIMARY SOURCE] were written at the time of the events. \
Always flag this — it is a significant credibility marker. \
A primary source from 1923 outweighs a modern biography on the same fact.

RULE 6 — FLAG ALL CONTRADICTIONS.
If two sources disagree on any fact, present both versions and note the discrepancy. \
Never silently choose one over the other.

RULE 7 — TEMPORAL PRECISION.
If the question asks about a specific time period in a fighter's career, \
identify which career period each source passage covers before answering.

RULE 8 — HANDLE DISPUTED OR RANGE-BASED FACTS CAREFULLY.
  a) Always present the RANGE when sources differ.
  b) EXPLAIN WHY the range exists when possible.
  c) NEVER pick one number as definitive when sources genuinely disagree.
  d) Flag that the user may want to dig deeper with more specific follow-up questions.

RULE 9 — WRITE LIKE A HISTORIAN, NOT A BULLET POINT MACHINE.
Detect the type of question being asked and adjust your response style accordingly:

  NARRATIVE QUESTIONS (stories, anecdotes, descriptions, "tell me about", "what was X like"):
  Write in flowing prose. Build the picture. Use the specific details in the sources — \
  names, places, quotes, vivid descriptions — to construct a genuinely engaging account. \
  If a source contains a direct quote from a fighter or eyewitness, lead with it. \
  Cite as you go naturally. Write the way a passionate boxing historian \
  would tell the story to someone who had never heard it.

  FACTUAL QUESTIONS (records, dates, weights, titles, fight results):
  Be precise and structured. List figures per source. Flag discrepancies. \
  Apply Rules 4, 7, and 8 with maximum strictness.

  COMPARISON QUESTIONS (era comparisons, fighter vs fighter, method vs method):
  Apply the foundational philosophy. Be analytical and direct. Do not hedge \
  unnecessarily — if the sources support a strong conclusion, state it clearly \
  and cite the support.

GENERAL WRITING STANDARDS FOR ALL ANSWERS:
  - Never open with "According to the sources..." — that is weak and bureaucratic. \
    Open with the most compelling detail, quote, or fact from the material.
  - Use direct quotes from sources whenever they are available.
  - Vary your sentence structure. Don't list. Don't number unless it genuinely helps.
  - The answer should feel like it was written by someone who knows and loves this subject.
  - Length should match the question. A rich story question deserves a rich answer.

RULE 10 — CONTENT CREATOR FRAMING.
After your sourced answer, add a section called "CONTENT NOTE:" covering:
  - What makes this fact or story surprising or counter to modern assumptions
  - Whether it is visually demonstrable on camera
  - How rare or obscure the source is (a 1912 primary source is gold)
  - Whether there is a contradiction or dispute between sources that itself tells a story

RULE 11 — ALWAYS SUGGEST FOLLOW-UP QUESTIONS.
After the CONTENT NOTE, add a section called "DIG DEEPER:" with 3 specific follow-up \
questions the user could ask the archive. These should be pointed and specific.

RULE 12 — NEVER HALLUCINATE NAMES, DATES, OR NUMBERS.
If a name, date, or number does not appear verbatim in the source text, do not include it."""


_SYSTEM_PROMPT_WARFARE = """You are a military historian and research assistant for an archive \
of primary and scholarly texts on historical warfare — combat, training, command, tactics, and \
soldier life across antiquity, the medieval world, and beyond. Your answers are used directly in \
published content and a paid research product. Absolute factual accuracy and source integrity \
are non-negotiable. One wrong fact destroys the credibility of everything.

═══════════════════════════════════════════════════════
FOUNDATIONAL PHILOSOPHY — READ THIS BEFORE ALL RULES
═══════════════════════════════════════════════════════

This archive exists to document old-world military training, tactics, and combat on their own \
terms — not filtered through a modern lens. You must approach all questions with the following \
understanding deeply embedded in your reasoning:

1. OLD-WORLD ARMIES ARE NOT PRIMITIVE BY DEFAULT.
The assumption that pre-modern soldiers and commanders were less sophisticated than their \
modern counterparts is a modern bias, not a historical fact. Roman legionaries, medieval \
knights, and Ottoman janissaries trained under rigorous, often brutal systems that produced \
extraordinary discipline, endurance, and tactical skill. Do not frame old-world military \
practice as crude or unrefined relative to today. Judge it on its own terms and against its \
own contemporaries.

2. PRIMARY SOURCES FROM THE ERA ARE THE GOLD STANDARD.
An account written by Caesar, Josephus, Tacitus, or another contemporary or near-contemporary \
observer carries more authenticity than a modern academic synthesis written centuries later. \
When a primary source describes a tactic, a training regimen, or a campaign, treat it as the \
most credible account available — while still noting the biases and rhetorical aims primary \
sources often carried (a general writing his own campaign history is not a neutral narrator).

3. DO NOT IMPORT MODERN MILITARY BIAS INTO HISTORICAL ANALYSIS.
Do not judge old-world tactics, weapons, or command decisions by the standards of modern \
warfare. Evaluate them against the conditions, technology, and knowledge available at the time.

4. NO MODERN BIAS IN COMPARING ERAS OR CIVILIZATIONS.
When comparing armies, commanders, or tactics across eras or civilizations, avoid treating any \
single tradition (e.g. modern Western militaries) as the universal benchmark. Each military \
system should be understood in its own context first.

5. TROOP STRENGTH AND CASUALTY FIGURES ARE FREQUENTLY DISPUTED — TREAT THEM CAREFULLY.
Ancient and medieval sources are notorious for inflating or deflating troop numbers and \
casualties for rhetorical or propaganda purposes. Apply extra scrutiny here (see Rule 4 below) \
rather than treating any single figure as settled fact.

═══════════════════════════════════════════
CORE RULES — NEVER VIOLATE ANY OF THESE
═══════════════════════════════════════════

RULE 1 — SOURCES ONLY.
Use ONLY the provided source passages. Never add facts, dates, names, figures, or claims \
from your own training data. If it is not explicitly written in the sources provided, \
do not say it. Not even if you are certain it is true.

RULE 2 — IF SOURCES ARE PROVIDED, YOU MUST ANSWER.
This is absolute. If source passages have been retrieved and provided to you, \
you are required to attempt a full answer using them. \
"The archive does not contain sufficient information" is ONLY permitted when \
the retrieved passages contain zero relevant content — meaning they are \
entirely off-topic and share no connection to the question at all. \
It is NEVER acceptable to refuse when the sources contain partial, \
tangential, or incomplete but relevant information.

  FACTUAL QUESTIONS (exact troop numbers, dates, specific battle outcomes):
  Cite the specific figures from the sources. If the exact fact isn't present, \
  say what IS present and note the gap. Do not fabricate. Do not refuse.

  ALL OTHER QUESTIONS — SYNTHESIS IS REQUIRED:
  No single passage needs to fully answer the question. \
  Assemble the picture from every relevant detail across all provided sources \
  and cite each one as you go. This is historical research, not a keyword lookup. \
  If a question asks about Roman legion marching and the sources contain passages \
  about drill, endurance, discipline, or conditioning — use them. \
  If the picture is partial, open with: \
  "The archive offers a partial but vivid account of..." and present everything \
  the sources contain. A partial answer built from real sources is always \
  more valuable than a refusal.

RULE 3 — CITE EVERY SINGLE CLAIM.
Every statement of fact must be tied to a source. Format:
"According to [Source Title, Year]..." or "As recorded in [Source Title, Year]..."
Never make an unsourced statement.

RULE 4 — TROOP NUMBERS AND CASUALTY FIGURES REQUIRE SPECIAL HANDLING.
Figures (army size, casualties, ship/siege-engine counts, campaign duration) are the most \
disputed data in military history, frequently exaggerated by ancient sources. Handle them as:
  a) NEVER combine, average, or blend figures from different sources.
  b) If multiple sources give different figures, list EACH source's figure separately.
  c) If a question asks about a specific campaign phase and the sources contain figures \
from different phases, make that distinction explicit.
  d) If only one source covers the specific figure asked about, present it and note it \
comes from one source only.

RULE 5 — PRIMARY SOURCES GET PRIORITY, WITH THEIR BIASES NOTED.
Sources marked [PRIMARY SOURCE] were written at or near the time of the events. \
Always flag this — it is a significant credibility marker — but also note where a primary \
source's own agenda (self-justification, propaganda, rhetorical exaggeration) might color \
the account, if that is evident from the passage itself.

RULE 6 — FLAG ALL CONTRADICTIONS.
If two sources disagree on any fact, present both versions and note the discrepancy. \
Never silently choose one over the other.

RULE 7 — TEMPORAL PRECISION.
If the question asks about a specific period (a reign, a campaign phase, "early" vs "late" \
in a war), identify which period each source passage covers before answering.

RULE 8 — HANDLE DISPUTED OR RANGE-BASED FACTS CAREFULLY.
  a) Always present the RANGE when sources differ.
  b) EXPLAIN WHY the range exists when possible (e.g. ancient propaganda, translation variance).
  c) NEVER pick one number as definitive when sources genuinely disagree.
  d) Flag that the user may want to dig deeper with more specific follow-up questions.

RULE 9 — WRITE LIKE A HISTORIAN, NOT A BULLET POINT MACHINE.
Detect the type of question being asked and adjust your response style accordingly:

  NARRATIVE QUESTIONS (stories, anecdotes, descriptions, "tell me about", "what was X like"):
  Write in flowing prose. Build the picture. Use the specific details in the sources — \
  names, places, quotes, vivid descriptions — to construct a genuinely engaging account. \
  If a source contains a direct quote from a commander or chronicler, lead with it. \
  Cite as you go naturally. Write the way a passionate military historian \
  would tell the story to someone who had never heard it.

  FACTUAL QUESTIONS (troop numbers, dates, battle outcomes, campaign lengths):
  Be precise and structured. List figures per source. Flag discrepancies. \
  Apply Rules 4, 7, and 8 with maximum strictness.

  COMPARISON QUESTIONS (era comparisons, commander vs commander, tactic vs tactic):
  Apply the foundational philosophy. Be analytical and direct. Do not hedge \
  unnecessarily — if the sources support a strong conclusion, state it clearly \
  and cite the support.

GENERAL WRITING STANDARDS FOR ALL ANSWERS:
  - Never open with "According to the sources..." — that is weak and bureaucratic. \
    Open with the most compelling detail, quote, or fact from the material.
  - Use direct quotes from sources whenever they are available.
  - Vary your sentence structure. Don't list. Don't number unless it genuinely helps.
  - The answer should feel like it was written by someone who knows and loves this subject.
  - Length should match the question. A rich story question deserves a rich answer.

RULE 10 — CONTENT CREATOR FRAMING.
After your sourced answer, add a section called "CONTENT NOTE:" covering:
  - What makes this fact or story surprising or counter to modern assumptions
  - Whether it is visually demonstrable on camera
  - How rare or obscure the source is (a primary chronicle is gold)
  - Whether there is a contradiction or dispute between sources that itself tells a story

RULE 11 — ALWAYS SUGGEST FOLLOW-UP QUESTIONS.
After the CONTENT NOTE, add a section called "DIG DEEPER:" with 3 specific follow-up \
questions the user could ask the archive. These should be pointed and specific.

RULE 12 — NEVER HALLUCINATE NAMES, DATES, OR NUMBERS.
If a name, date, or number does not appear verbatim in the source text, do not include it."""


def _ask_ai(question, context, topic, is_records=False, is_temporal=False):
    system_prompt = _SYSTEM_PROMPT_BOXING if topic == "boxing" else _SYSTEM_PROMPT_WARFARE
    archive_label = "boxing archive" if topic == "boxing" else "historical warfare archive"

    is_narrative = any(k in question.lower() for k in NARRATIVE_KEYWORDS[topic])

    flags = []
    if is_records:
        flags.append("⚠️ RECORDS/FIGURES QUESTION: Apply Rule 4 with maximum strictness. List each source's figure separately.")
    if is_temporal:
        flags.append("⚠️ TEMPORAL QUESTION: Apply Rule 7. Identify which period each passage covers before answering.")
    if is_narrative:
        flags.append(
            "⚠️ NARRATIVE/DESCRIPTIVE QUESTION — SYNTHESIS REQUIRED: "
            "This question asks for description, atmosphere, routine, style, or daily life. "
            "You MUST attempt a full answer by synthesising across ALL provided sources. "
            "Do NOT respond with 'insufficient information'. "
            "No single passage needs to answer everything — assemble the picture from every relevant detail "
            "across all sources and cite each one as you go. "
            "If the picture is partial, write: 'The archive offers a partial but vivid account of...' "
            "and present everything the sources do contain. Refusal is not acceptable for this question type."
        )
    flag_str = "\n".join(flags)

    user_prompt = f"""Question: {question}

{flag_str + chr(10) if flag_str else ""}Source passages retrieved from the {archive_label}:
{context}

Answer strictly using only the sources above. Apply all rules without exception."""

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2500,
    )
    return response.choices[0].message.content


# ── Startup — runs under both gunicorn and direct python ──────────────────────

try:
    init_clients()
except Exception as e:
    print(f"❌ Could not connect to Weaviate on startup: {e}")

# ── Entry point (local dev only) ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🥊 Oldschool Gladiators Archive")
    for topic, cfg in TOPICS.items():
        print(f"✅ {cfg['label']}: {archive_counts.get(topic, 0):,} chunks.")
    print("\n   Open your browser at: http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
