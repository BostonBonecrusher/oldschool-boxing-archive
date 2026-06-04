"""
03_search.py
------------
Search your boxing archive. Ask any question and get sourced answers
pulled directly from your PDFs.

Usage:
    python 03_search.py

Type a question at the prompt. Type 'quit' to exit.

Optional — search for a specific keyword:
    python 03_search.py --keyword "Sugar Ray Robinson"
"""

import os
import sys
import re
import argparse
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery, HybridFusion, Filter
from openai import OpenAI

load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COLLECTION_NAME = "BoxingChunk"

# Default chunks to retrieve
TOP_K_DEFAULT = 10
# Extra chunks for records/stats questions
TOP_K_RECORDS = 15

# Keywords that indicate a narrative / descriptive question requiring synthesis
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

# Keywords that indicate a records or stats question
RECORDS_KEYWORDS = [
    'record', 'fight', 'won', 'lost', 'draw', 'knockout', 'ko', 'total fights',
    'how many', 'wins', 'losses', 'undefeated', 'career', 'times', 'weight',
    'pounds', 'lbs', 'champion', 'title', 'reign', 'held', 'defended'
]

# Keywords that indicate a time-specific question
TEMPORAL_KEYWORDS = [
    'early career', 'late career', 'end of career', 'beginning', 'later',
    'prime', 'young', 'old', 'retire', 'final', 'last fight', 'first fight',
    'at the time', 'during', 'before', 'after', 'by the time'
]


# ─────────────────────────────────────────────────────────────
# PHASE 2 — SMART FILTERING USING AI TAGS
# ─────────────────────────────────────────────────────────────

# Words that look capitalised in questions but aren't fighter names.
# Includes common book-title words so they don't get treated as fighters.
_NAME_STOPWORDS = {
    'The', 'What', 'How', 'Who', 'When', 'Where', 'Why', 'Did', 'Was', 'Is',
    'Tell', 'Can', 'Could', 'Would', 'Should', 'In', 'At', 'On', 'By', 'Of',
    'His', 'Her', 'Their', 'He', 'She', 'They', 'It', 'And', 'Or', 'But',
    'For', 'With', 'About', 'From', 'Between', 'During', 'After', 'Before',
    'Old', 'New', 'First', 'Last', 'Early', 'Late', 'American', 'British',
    'World', 'Title', 'Championship', 'Fight', 'Fights', 'Bout', 'Round', 'Boxing',
    'Career', 'Era', 'History', 'Prime', 'Final', 'Known', 'Famous', 'Great',
    # Common book-title words — should never be treated as fighter names
    'Nights', 'Night', 'Stateside', 'Scraps', 'Scrapes', 'Scuffles', 'Tales',
    'Stories', 'Story', 'Ring', 'Corner', 'Science', 'Art', 'Guide', 'Methods',
    'Secrets', 'Memoirs', 'Years', 'Days', 'Life', 'Lives', 'York', 'London',
    'Chicago', 'Gazette', 'Journal', 'Record', 'Press', 'Tribune', 'Times',
    'Arc', 'Gladiators', 'Angels', 'Sweet', 'Craft', 'Book', 'Vol',
}

# Trigger words that signal the user is referencing a specific source/book
_SOURCE_TRIGGERS = [
    'in ', 'from ', 'check ', 'according to ', 'it says in ', 'states in ',
    'mentioned in ', 'look in ', 'what does ', 'that book', 'the book',
]


def extract_source_reference(question):
    """
    Detect if the user is referencing a specific book or source by name.
    Returns a list of distinctive lowercase keywords from the title reference
    that can be used to filter by source_file or title.
    E.g. "check stateside scraps" → ['stateside', 'scraps']
    """
    q_lower = question.lower()
    found = []

    # Scan for trigger phrases and grab the words that follow
    for trigger in _SOURCE_TRIGGERS:
        idx = q_lower.find(trigger)
        if idx == -1:
            continue
        fragment = question[idx + len(trigger):idx + len(trigger) + 80]
        words = re.findall(r'[A-Za-z]{3,}', fragment)
        for w in words:
            wl = w.lower()
            # Keep words that look like title content — not question stopwords
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


def build_source_filter(title_words):
    """
    Build a Weaviate filter that checks title and source_file for any of
    the given title keywords. Returns None if no useful words provided.
    """
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

# Keywords that signal specific content types
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


def extract_fighter_names(question):
    """
    Extract likely fighter names from a question.
    Returns a list of last-name tokens (e.g. ['Dempsey', 'Firpo']).
    Grabs capitalised words that aren't question/stopwords and are 3+ chars.
    """
    tokens = re.findall(r"[A-Z][a-z]{2,}", question)
    return [t for t in tokens if t not in _NAME_STOPWORDS]


def detect_content_intent(question):
    """
    Return a set of content-type intents detected in the question.
    E.g. {'fight_account', 'quotes'}
    """
    q = question.lower()
    detected = set()
    for intent, keywords in _INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            detected.add(intent)
    return detected


def build_fighter_filter(names):
    """
    Build a Weaviate Filter that matches any chunk mentioning at least
    one of the given fighter last-names in its fighters_mentioned field.
    """
    if not names:
        return None
    conditions = [
        Filter.by_property("fighters_mentioned").like(f"*{name}*")
        for name in names
        if len(name) > 2
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


def build_intent_filter(intent):
    """
    Build a Weaviate Filter for the detected content intent(s).
    Uses OR logic so any matching content type qualifies.
    """
    field_map = {
        "fight_account":   ("contains_fight_account",    True),
        "biographical":    ("contains_biographical_info", True),
        "training":        ("contains_training_methods",  True),
        "quotes":          ("quotes_present",             True),
        "controversy":     ("controversy_present",        True),
        "statistics":      ("has_statistics",             True),
    }
    conditions = [
        Filter.by_property(field).equal(val)
        for key, (field, val) in field_map.items()
        if key in intent
    ]
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else Filter.any_of(conditions)


# Properties to return from every hybrid search
_RETURN_PROPS = [
    "content", "source_file", "title", "author",
    "year_published", "document_type", "era", "rules_era",
    "discipline", "is_primary_source", "subject_fighter",
    "geographic_focus", "page_number", "chunk_index", "ocr_used",
    "fighters_mentioned", "weight_class_focus",
]


def _run_hybrid(collection, query, top_k, filters=None):
    """Core hybrid search, optionally filtered."""
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
    """Merge multiple result lists, deduplicating by first 100 chars of content."""
    seen, merged = set(), []
    for lst in lists:
        for chunk in lst:
            key = chunk.properties.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)
    return merged


def search_phase2(collection, query, top_k=None):
    """
    Phase 2 enhanced search.

    Strategy:
    1. Standard hybrid search (always runs — the reliable baseline).
    2. Fighter-name-filtered search (if names detected in question).
    3. Content-intent-filtered search (if question type detected).

    Filtered results are placed FIRST so the AI sees the most targeted
    passages at the top of its context. Standard results fill in the rest.
    Deduplication ensures no chunk appears twice.
    """
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records_question(query) else TOP_K_DEFAULT

    fighter_names  = extract_fighter_names(query)
    intent         = detect_content_intent(query)
    source_refs    = extract_source_reference(query)

    result_sets = []

    # ── Source title filter (highest priority — user named a specific book) ──
    source_filter = build_source_filter(source_refs)
    if source_filter is not None:
        r = _run_hybrid(collection, query, top_k, filters=source_filter)
        result_sets.append(r)

    # ── Fighter name + intent filters ─────────────────────────
    fighter_filter = build_fighter_filter(fighter_names)
    intent_filter  = build_intent_filter(intent)

    if fighter_filter is not None:
        # Fighter name filter alone
        r = _run_hybrid(collection, query, top_k, filters=fighter_filter)
        result_sets.append(r)

        # Fighter + intent combined (most targeted)
        if intent_filter is not None:
            combined = Filter.all_of([fighter_filter, intent_filter])
            r2 = _run_hybrid(collection, query, top_k, filters=combined)
            result_sets.insert(0, r2)  # highest priority

    elif intent_filter is not None:
        r = _run_hybrid(collection, query, top_k, filters=intent_filter)
        result_sets.append(r)

    # ── Baseline search (always) ───────────────────────────────
    result_sets.append(_run_hybrid(collection, query, top_k))

    merged = _merge_unique(result_sets)

    # Cap at a sensible maximum
    max_chunks = top_k + 8
    return merged[:max_chunks]


def is_records_question(question):
    """Detect if the question is about fight records or statistics."""
    q = question.lower()
    return any(k in q for k in RECORDS_KEYWORDS)


def is_temporal_question(question):
    """Detect if the question references a specific career period."""
    q = question.lower()
    return any(k in q for k in TEMPORAL_KEYWORDS)


def is_narrative_question(question):
    """Detect if the question asks for description, synthesis, or camp/lifestyle detail."""
    q = question.lower()
    return any(k in q for k in NARRATIVE_KEYWORDS)


def search_archive(collection, query, top_k=None):
    """
    Legacy direct hybrid search — used by the expand/keyword flow.
    For normal questions use search_phase2() which is smarter.
    """
    if top_k is None:
        top_k = TOP_K_RECORDS if is_records_question(query) else TOP_K_DEFAULT
    return _run_hybrid(collection, query, top_k)


def search_archive_expanded(collection, query):
    """
    For temporal or records questions, run a secondary keyword-focused pass
    and merge with Phase 2 results. Kept for backward compatibility.
    """
    results1 = search_phase2(collection, query)

    words = query.lower().split()
    secondary = " ".join([w for w in query.split() if w[0].isupper()] +
                         [w for w in words if w in RECORDS_KEYWORDS])
    if not secondary.strip():
        return results1

    results2 = _run_hybrid(collection, secondary, top_k=8)
    return _merge_unique([results1, results2])


def build_context(chunks):
    """
    Format retrieved chunks into a context block for the AI.
    Each source includes its full citation so the AI can reference it properly.
    Chunk index is included so the AI knows position within the document.
    """
    context_parts = []
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

        context_parts.append(f"[Source {i}: {citation}]\n{content}")

    return "\n\n---\n\n".join(context_parts)


def ask_ai(openai_client, question, context):
    """
    Send the question + retrieved context to GPT-4o and get a sourced answer.
    Strict no-hallucination and records-accuracy rules enforced.
    """
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
  b) If multiple sources give different figures, list EACH source's figure separately:
     "According to [Source A], Greb's record was 262 fights. According to [Source B], \
the figure given is 299. These sources differ — both are presented as recorded."
  c) If a question asks about a specific career period (early, late, prime, end of career) \
and the sources contain figures from different periods, make that distinction explicit. \
Do not present a late-career weight as an answer to a question about peak fighting weight.
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
identify which career period each source passage covers before answering. \
A passage about Greb's fighting weight in 1920 is not the same as his weight \
in 1926. Make this distinction clear.

RULE 8 — HANDLE DISPUTED OR RANGE-BASED FACTS CAREFULLY.
Some facts in boxing history are genuinely disputed — Hall of Fame counts, total fight records, \
exact weights, knockout figures. These disputes exist because different sources use different \
criteria, different time periods, or different definitions. Handle them as follows:
  a) Always present the RANGE when sources differ: \
"Sources in the archive give figures ranging from X to Y depending on the criteria applied."
  b) EXPLAIN WHY the range exists when possible — e.g. Hall of Fame status is retrospective \
and applied differently by different authors; fight records vary because some bouts were \
not formally recorded; weights vary across a career.
  c) NEVER pick one number as definitive when sources genuinely disagree.
  d) Flag that the user may want to dig deeper with more specific follow-up questions.

RULE 9 — WRITE LIKE A HISTORIAN, NOT A BULLET POINT MACHINE.
This is critical. Strict sourcing does not mean dry, listified, or cautious writing. \
A great historian cites every claim AND writes with depth, narrative, and vivid detail. \
You must do both simultaneously.

Detect the type of question being asked and adjust your response style accordingly:

  NARRATIVE QUESTIONS (stories, anecdotes, descriptions, "tell me about", "what was X like"):
  Write in flowing prose. Build the picture. Use the specific details in the sources — \
  names, places, quotes, vivid descriptions — to construct a genuinely engaging account. \
  If a source says Langford "walked through punches like they were raindrops", use that. \
  If a source contains a direct quote from a fighter or eyewitness, lead with it. \
  Cite as you go naturally — "As James Fair recorded in Give Him To The Angels..." — \
  not as a footnote after a dry summary. Write the way a passionate boxing historian \
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
  - Use direct quotes from sources whenever they are available — they are more \
    powerful than paraphrase.
  - Vary your sentence structure. Don't list. Don't number unless it genuinely helps.
  - The answer should feel like it was written by someone who knows and loves \
    this subject — because the archive does.
  - Length should match the question. A rich story question deserves a rich answer. \
    A simple factual question deserves a tight precise one.

RULE 10 — CONTENT CREATOR FRAMING.
After your sourced answer, add a section called "CONTENT NOTE:" covering:
  - What makes this fact or story surprising or counter to modern assumptions
  - Whether it is visually demonstrable on camera
  - How rare or obscure the source is (a 1912 primary source is gold)
  - Whether there is a contradiction or dispute between sources that itself tells a story
  - Whether the range of uncertainty (if any) is itself interesting content

RULE 11 — ALWAYS SUGGEST FOLLOW-UP QUESTIONS.
After the CONTENT NOTE, add a section called "DIG DEEPER:" with 3 specific follow-up \
questions the user could ask the archive to get more detail, resolve a dispute, or \
uncover a related fact. These should be pointed and specific — not generic. \
For example, instead of "ask about Greb's career", suggest \
"Ask: which specific fighters did Greb beat that are listed as Hall of Famers in [Source]?" \
This turns every answer into a research thread and helps the user build a complete picture.

RULE 12 — NEVER HALLUCINATE NAMES, DATES, OR NUMBERS.
If a name, date, or number does not appear verbatim in the source text, do not include it."""

    # Flag question type so the AI applies the correct mode
    flags = []
    if is_records_question(question):
        flags.append("⚠️ RECORDS QUESTION: Apply Rule 4 with maximum strictness. List each source's figure separately.")
    if is_temporal_question(question):
        flags.append("⚠️ TEMPORAL QUESTION: Apply Rule 7. Identify which career period each passage covers before answering.")
    if is_narrative_question(question):
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
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=2500,
    )
    return response.choices[0].message.content


def print_sources(chunks, expand=None):
    """
    Print source citations with previews.
    expand = None      → show 350 char preview for all
    expand = [1,3]     → show full text for sources 1 and 3, preview for rest
    expand = 'all'     → show full text for every source
    """
    print("\n" + "─" * 55)
    print("📄 SOURCES RETRIEVED FROM ARCHIVE:")
    print("─" * 55)
    print("   Tip: type 'expand 2' or 'expand 1 3 5' or 'expand all' to read full passages\n")

    for i, chunk in enumerate(chunks, 1):
        props    = chunk.properties
        title    = props.get("title") or props.get("source_file", "Unknown")
        author   = props.get("author", "")
        year     = props.get("year_published", 0)
        era      = props.get("era", "")
        doc_type = props.get("document_type", "")
        primary  = props.get("is_primary_source", False)
        fighter  = props.get("subject_fighter", "")
        geo      = props.get("geographic_focus", "")
        score    = chunk.metadata.score if chunk.metadata else 0
        ocr      = " [OCR]" if props.get("ocr_used") else ""
        content  = props.get("content", "")

        # Citation line
        citation = title
        if author:  citation += f" — {author}"
        if year:    citation += f" ({year})"

        # Tags line
        tags = []
        if primary:   tags.append("PRIMARY SOURCE")
        if doc_type:  tags.append(doc_type)
        if era:       tags.append(era)
        if geo:       tags.append(geo)
        if fighter:   tags.append(f"subject: {fighter}")
        tag_str = "  |  ".join(tags)

        # Decide whether to show full text or preview
        show_full = (expand == 'all') or (isinstance(expand, list) and i in expand)

        print(f"\n[{i}] {citation}{ocr}")
        if tag_str:
            print(f"     {tag_str}")
        print(f"     Relevance: {score:.3f}")

        if show_full:
            print(f"\n{'─'*40}")
            print(f"FULL PASSAGE [{i}]:")
            print(f"{'─'*40}")
            print(content)
            print(f"{'─'*40}")
        else:
            preview = content[:350] + ("..." if len(content) > 350 else "")
            print(f"     \"{preview}\"")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", help="Search by keyword instead of AI question")
    parser.add_argument("--show-sources", action="store_true", default=True,
                        help="Show source passages after the answer (default: on)")
    parser.add_argument("--no-sources", action="store_true",
                        help="Hide source passages")
    args = parser.parse_args()

    show_sources = args.show_sources and not args.no_sources

    # ── Check config ──
    if not WEAVIATE_URL or "YOUR-CLUSTER" in WEAVIATE_URL:
        print("❌ WEAVIATE_URL not set in .env file.")
        sys.exit(1)

    # ── Connect ──
    print("🔌 Connecting to boxing archive...")
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
        )
        collection = client.collections.get(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Could not connect: {e}")
        sys.exit(1)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # Check how many chunks are in the archive
    try:
        count = collection.aggregate.over_all(total_count=True).total_count
        print(f"✅ Connected! Archive contains {count:,} chunks from your PDFs.\n")
    except Exception:
        print("✅ Connected!\n")

    # ── One-off keyword search ──
    if args.keyword:
        print(f"🔍 Searching for: '{args.keyword}'\n")
        chunks = search_archive(collection, args.keyword)
        if not chunks:
            print("No results found.")
        else:
            print_sources(chunks)
        client.close()
        return

    # ── Interactive mode ──
    print("=" * 55)
    print("  🥊  BOXING ARCHIVE — OLDSCHOOL GLADIATORS")
    print("=" * 55)
    print("Ask any boxing history question. Type 'quit' to exit.")
    print("After any answer: 'expand 2' / 'expand 1 3 5' / 'expand all'")
    print("Type 'sources on' or 'sources off' to toggle source display.\n")

    last_chunks = []  # Store last result so expand commands work

    while True:
        try:
            question = input("❓ Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "sources on":
            show_sources = True
            print("📄 Source display: ON\n")
            continue

        if question.lower() == "sources off":
            show_sources = False
            print("📄 Source display: OFF\n")
            continue

        # ── Expand command ──
        if question.lower().startswith("expand"):
            if not last_chunks:
                print("❌ No previous search to expand. Ask a question first.\n")
                continue
            parts = question.lower().split()
            if "all" in parts:
                print_sources(last_chunks, expand='all')
            else:
                try:
                    nums = [int(p) for p in parts[1:] if p.isdigit()]
                    if nums:
                        print_sources(last_chunks, expand=nums)
                    else:
                        print("Usage: expand 2  /  expand 1 3 5  /  expand all\n")
                except ValueError:
                    print("Usage: expand 2  /  expand 1 3 5  /  expand all\n")
            continue

        print("\n🔍 Searching archive...")

        try:
            # Step 1: Find relevant chunks (Phase 2 enhanced)
            fighter_names = extract_fighter_names(question)
            intent        = detect_content_intent(question)
            source_refs   = extract_source_reference(question)

            if is_records_question(question) or is_temporal_question(question):
                chunks = search_archive_expanded(collection, question)
                if is_records_question(question):
                    print("   📊 Records question — retrieving extra sources...")
                if is_temporal_question(question):
                    print("   🕐 Time-period question — expanding search...")
            else:
                chunks = search_phase2(collection, question)

            if fighter_names:
                print(f"   🥊 Fighters detected: {', '.join(fighter_names)}")
            if intent:
                print(f"   🏷️  Content filters: {', '.join(sorted(intent))}")
            if source_refs:
                print(f"   📖 Source reference: {', '.join(source_refs)}")

            if not chunks:
                print("❌ No relevant passages found in the archive for that question.\n")
                continue

            last_chunks = chunks  # Save for expand commands

            # Step 2: Build context
            context = build_context(chunks)

            # Step 3: Get AI answer
            print("🤖 Generating answer...\n")
            answer = ask_ai(openai_client, question, context)

            # Step 4: Display
            print("─" * 55)
            print("💬 ANSWER:")
            print("─" * 55)
            print(answer)

            if show_sources:
                print_sources(chunks)

            print()

        except Exception as e:
            print(f"❌ Error: {e}\n")

    client.close()


if __name__ == "__main__":
    main()
