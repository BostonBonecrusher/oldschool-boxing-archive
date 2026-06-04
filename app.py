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
from functools import wraps
from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter
from openai import OpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
APP_PASSWORD     = os.getenv("APP_PASSWORD", "gladiators")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-before-deploying")
COLLECTION_NAME  = "BoxingChunk"

TOP_K_DEFAULT = 10
TOP_K_RECORDS = 15

NARRATIVE_KEYWORDS = [
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
]

RECORDS_KEYWORDS = [
    'record', 'fight', 'won', 'lost', 'draw', 'knockout', 'ko', 'total fights',
    'how many', 'wins', 'losses', 'undefeated', 'career', 'times', 'weight',
    'pounds', 'lbs', 'champion', 'title', 'reign', 'held', 'defended'
]
TEMPORAL_KEYWORDS = [
    'early career', 'late career', 'end of career', 'beginning', 'later',
    'prime', 'young', 'old', 'retire', 'final', 'last fight', 'first fight',
    'at the time', 'during', 'before', 'after', 'by the time'
]

# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ── Global clients ────────────────────────────────────────────────────────────

weaviate_client = None
collection      = None
openai_client   = None
archive_count   = 0


def init_clients():
    global weaviate_client, collection, openai_client, archive_count
    weaviate_client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
        headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
    )
    collection = weaviate_client.collections.get(COLLECTION_NAME)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        archive_count = collection.aggregate.over_all(total_count=True).total_count
    except Exception:
        archive_count = 0


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
    return render_template("index.html", chunk_count=f"{archive_count:,}")


@app.route("/stitch", methods=["POST"])
@login_required
def stitch():
    """
    Given a source_file and chunk_index, fetch the surrounding chunks
    and return them stitched together as a continuous passage.
    """
    data        = request.get_json()
    source_file = (data.get("source_file") or "").strip()
    chunk_index = int(data.get("chunk_index", 0))
    radius      = int(data.get("radius", 2))   # chunks before + after to fetch

    if not source_file:
        return jsonify({"error": "source_file required"}), 400

    try:
        start = max(0, chunk_index - radius)
        end   = chunk_index + radius

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


@app.route("/search", methods=["POST"])
@login_required
def search():
    data = request.get_json()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        is_records  = any(k in question.lower() for k in RECORDS_KEYWORDS)
        is_temporal = any(k in question.lower() for k in TEMPORAL_KEYWORDS)

        # Phase 2 detection (for response metadata)
        fighter_names = _extract_fighter_names(question)
        intent        = _detect_content_intent(question)

        # Retrieve chunks
        if is_records or is_temporal:
            chunks = _search_expanded(question)
        else:
            chunks = _search_phase2(question)

        if not chunks:
            return jsonify({"error": "No relevant passages found in the archive for that question."})

        context = _build_context(chunks)
        answer  = _ask_ai(question, context, is_records, is_temporal)

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
                "subject": props.get("subject_fighter", ""),
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
                "fighters":      fighter_names,
                "intent":        sorted(intent),
            },
        })

    except Exception as e:
        return jsonify({"error": f"Search error: {str(e)}"}), 500


# ── Search helpers ────────────────────────────────────────────────────────────

# ── Phase 2 helpers ───────────────────────────────────────────────────────────

_NAME_STOPWORDS = {
    'The', 'What', 'How', 'Who', 'When', 'Where', 'Why', 'Did', 'Was', 'Is',
    'Tell', 'Can', 'Could', 'Would', 'Should', 'In', 'At', 'On', 'By', 'Of',
    'His', 'Her', 'Their', 'He', 'She', 'They', 'It', 'And', 'Or', 'But',
    'For', 'With', 'About', 'From', 'Between', 'During', 'After', 'Before',
    'Old', 'New', 'First', 'Last', 'Early', 'Late', 'American', 'British',
    'World', 'Title', 'Championship', 'Fight', 'Fights', 'Bout', 'Round', 'Boxing',
    'Career', 'Era', 'History', 'Prime', 'Final', 'Known', 'Famous', 'Great',
    'Nights', 'Night', 'Stateside', 'Scraps', 'Scrapes', 'Scuffles', 'Tales',
    'Stories', 'Story', 'Ring', 'Corner', 'Science', 'Art', 'Guide', 'Methods',
    'Secrets', 'Memoirs', 'Years', 'Days', 'Life', 'Lives', 'York', 'London',
    'Chicago', 'Gazette', 'Journal', 'Record', 'Press', 'Tribune', 'Times',
    'Arc', 'Gladiators', 'Angels', 'Sweet', 'Craft', 'Book', 'Vol',
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

_INTENT_MAP = {
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

_RETURN_PROPS = [
    "content", "source_file", "title", "author",
    "year_published", "document_type", "era", "rules_era",
    "discipline", "is_primary_source", "subject_fighter",
    "geographic_focus", "page_number", "chunk_index", "ocr_used",
    "fighters_mentioned", "weight_class_focus",
]


def _extract_fighter_names(question):
    tokens = re.findall(r"[A-Z][a-z]{2,}", question)
    return [t for t in tokens if t not in _NAME_STOPWORDS]


def _detect_content_intent(question):
    q = question.lower()
    detected = set()
    for intent, keywords in _INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            detected.add(intent)
    return detected


def _build_fighter_filter(names):
    if not names:
        return None
    conditions = [
        Filter.by_property("fighters_mentioned").like(f"*{name}*")
        for name in names if len(name) > 2
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


def _build_intent_filter(intent):
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
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


def _run_hybrid(query, top_k, filters=None):
    kwargs = dict(
        query=query,
        limit=top_k,
        fusion_type=HybridFusion.RELATIVE_SCORE,
        return_metadata=MetadataQuery(score=True),
        return_properties=_RETURN_PROPS,
    )
    if filters is not None:
        kwargs["filters"] = filters
    return collection.query.hybrid(**kwargs).objects


def _merge_unique(lists):
    seen, merged = set(), []
    for lst in lists:
        for chunk in lst:
            key = chunk.properties.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged


def _search_archive(query, top_k=None):
    """Legacy direct search — used internally and for fallback."""
    is_records = any(k in query.lower() for k in RECORDS_KEYWORDS)
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records else TOP_K_DEFAULT
    return _run_hybrid(query, top_k)


def _search_phase2(query, top_k=None):
    """
    Phase 2 enhanced search.
    Runs fighter-name and content-intent filtered passes first,
    then falls back to standard hybrid to ensure full coverage.
    """
    is_records = any(k in query.lower() for k in RECORDS_KEYWORDS)
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records else TOP_K_DEFAULT

    fighter_names  = _extract_fighter_names(query)
    intent         = _detect_content_intent(query)
    source_refs    = _extract_source_reference(query)

    result_sets = []

    # Source title filter — highest priority
    source_filter = _build_source_filter(source_refs)
    if source_filter is not None:
        result_sets.append(_run_hybrid(query, top_k, filters=source_filter))

    fighter_filter = _build_fighter_filter(fighter_names)
    intent_filter  = _build_intent_filter(intent)

    if fighter_filter is not None:
        result_sets.append(_run_hybrid(query, top_k, filters=fighter_filter))
        if intent_filter is not None:
            combined = Filter.all_of([fighter_filter, intent_filter])
            result_sets.insert(0, _run_hybrid(query, top_k, filters=combined))
    elif intent_filter is not None:
        result_sets.append(_run_hybrid(query, top_k, filters=intent_filter))

    # Always include baseline
    result_sets.append(_run_hybrid(query, top_k))

    merged = _merge_unique(result_sets)
    return merged[:top_k + 8]


def _search_expanded(query):
    results1 = _search_phase2(query)
    words = query.lower().split()
    secondary = " ".join(
        [w for w in query.split() if w[0].isupper()] +
        [w for w in words if w in RECORDS_KEYWORDS]
    )
    if not secondary.strip():
        return results1
    results2 = _run_hybrid(secondary, top_k=8)
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


def _ask_ai(question, context, is_records=False, is_temporal=False):
    system_prompt = """You are a boxing historian and research assistant for a YouTube channel \
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

    is_narrative = any(k in question.lower() for k in NARRATIVE_KEYWORDS)

    flags = []
    if is_records:
        flags.append("⚠️ RECORDS QUESTION: Apply Rule 4 with maximum strictness. List each source's figure separately.")
    if is_temporal:
        flags.append("⚠️ TEMPORAL QUESTION: Apply Rule 7. Identify which career period each passage covers before answering.")
    if is_narrative:
        flags.append(
            "⚠️ NARRATIVE/DESCRIPTIVE QUESTION — SYNTHESIS REQUIRED: "
            "This question asks for description, atmosphere, routine, style, or camp life. "
            "You MUST attempt a full answer by synthesising across ALL provided sources. "
            "Do NOT respond with 'insufficient information'. "
            "No single passage needs to answer everything — assemble the picture from every relevant detail "
            "across all sources and cite each one as you go. "
            "If the picture is partial, write: 'The archive offers a partial but vivid account of...' "
            "and present everything the sources do contain. Refusal is not acceptable for this question type."
        )
    flag_str = "\n".join(flags)

    user_prompt = f"""Question: {question}

{flag_str + chr(10) if flag_str else ""}Source passages retrieved from the boxing archive:
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🥊 Oldschool Gladiators Boxing Archive")
    print("   Connecting to archive...\n")
    try:
        init_clients()
        print(f"✅ Connected! Archive contains {archive_count:,} chunks.\n")
        print("   Open your browser at: http://localhost:5000\n")
        app.run(debug=False, host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"❌ Could not start: {e}")
        print("\nCheck that your .env file is filled in correctly.")
