"""
02_ingest_warfare.py
---------------------
Ingestion script for the Historical Warfare archive (Roman/Greek/Byzantine/
Ottoman/Arab/conquest history — the "roman empire" book folder). Mirrors
02_ingest.py's pipeline (OCR fallback, chunking, batch upload, resumable
log) but:

  1. Only processes files listed in warfare_manifest.txt — NOT every file
     in the folder (that folder has 187 PDFs; only ~60 are in scope).
  2. Uses a hand-verified metadata table (FILE_METADATA below) for those
     60 curated files instead of pure filename-keyword guessing, since
     author/era/conflict attribution matters for citation accuracy in a
     "sourced answers" database. Any file NOT in that table (e.g. one
     added to the manifest later) falls back to generic keyword/regex
     classifiers, same spirit as the boxing archive's approach.

Saves progress as it goes — if it stops, run it again and it picks up
where it left off.

Usage:
    python 02_ingest_warfare.py

Re-process everything from scratch:
    python 02_ingest_warfare.py --reset
"""

import os
import sys
import json
import time
import gc
import argparse
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
import pdfplumber
import pytesseract
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image
import weaviate
from weaviate.classes.init import Auth
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
load_dotenv()

WEAVIATE_URL      = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY  = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
PDF_FOLDER        = os.getenv("WARFARE_PDF_FOLDER", r"C:\Users\tatte\OneDrive\roman empire")
MANIFEST_FILE     = os.getenv("WARFARE_MANIFEST_FILE", str(Path(PDF_FOLDER) / "warfare_manifest.txt"))
TESSERACT_PATH    = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH      = os.getenv("POPPLER_PATH", None)
CHUNK_SIZE        = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP     = int(os.getenv("CHUNK_OVERLAP", "200"))

COLLECTION_NAME = "HistoricalWarfareChunk"
LOG_FILE        = "ingestion_log_warfare.json"
OCR_THRESHOLD   = 100  # avg chars/page below this triggers OCR


# ─────────────────────────────────────────────────────────────
# Manifest (curated subset of the folder to actually ingest)
# ─────────────────────────────────────────────────────────────

def load_manifest(manifest_path):
    """
    Read warfare_manifest.txt — one filename per line, '#' comments and
    blank lines ignored. Returns a set of exact filenames.
    """
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest file not found: {manifest_path}")
        sys.exit(1)
    names = set()
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line)
    return names


# ─────────────────────────────────────────────────────────────
# Progress log
# ─────────────────────────────────────────────────────────────

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": {}, "started_at": datetime.now().isoformat()}

def save_log(log):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


# ─────────────────────────────────────────────────────────────
# PDF extraction (identical pipeline to the boxing archive)
# ─────────────────────────────────────────────────────────────

def _is_reversed_token(tok):
    """A token like 'semaG' or 'cilbuP' -- starts lowercase, ends uppercase,
    core is alphabetic. This is essentially impossible in normal English/
    Latin prose, but is exactly what falls out when a Title Case phrase
    ('Games', 'Public') has its whole character order reversed."""
    core = tok.strip(",.;:()-–—\"'")
    if len(core) < 3:
        return False
    if not core[0].isalpha() or not core[-1].isalpha():
        return False
    return core[0].islower() and core[-1].isupper()

def _is_bridgeable_glue(tok):
    """Short non-alphabetic or roman-numeral-style tokens that can sit
    *inside* a reversed run without being individually flagged (e.g. the
    page number, or 'II' from 'Part II')."""
    core = tok.strip(",.;:()-–—\"'")
    if core == "":
        return True
    if core.isdigit():
        return True
    if core.isalpha() and core.isupper() and len(core) <= 5:
        return True
    return False

def fix_reversed_header_runs(text):
    """
    Repairs a recurring pdfplumber extraction artifact found in some PDFs
    (confirmed in TheHistoryOfAncientRome.pdf, ~78 pages): a running
    header/footer text element (chapter title, photo credit line, etc.)
    gets extracted with its entire character order reversed -- e.g.
    "cilbuP tnemniatretnE" instead of "Public Entertainment". The source
    PDF itself is clean (verified directly with pdftotext); this is
    introduced by pdfplumber's own word/line clustering for that specific
    text run, likely tied to how it's positioned/rotated in the page.

    Detection: a run of 3+ consecutive tokens that each start lowercase
    and end uppercase essentially never happens in real prose, but is
    exactly what you get when a normal Title Case phrase is reversed
    character-by-character. Fix: reverse the whole matched span (as one
    string) back to its original character order.
    """
    if not text:
        return text
    tokens = text.split(' ')
    n = len(tokens)
    flagged = [_is_reversed_token(t) for t in tokens]
    i = 0
    out = []
    while i < n:
        if flagged[i]:
            j = i
            last_flagged = i
            flagged_count = 1
            while j + 1 < n:
                nxt = j + 1
                if flagged[nxt]:
                    last_flagged = nxt
                    flagged_count += 1
                    j = nxt
                elif _is_bridgeable_glue(tokens[nxt]):
                    j = nxt
                else:
                    break
            if flagged_count >= 3:
                span = tokens[i:last_flagged + 1]
                out.append(' '.join(span)[::-1])
                i = last_flagged + 1
                continue
        out.append(tokens[i])
        i += 1
    return ' '.join(out)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', ' ', text)
    text = fix_reversed_header_runs(text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    return text.strip()

def text_looks_real(text):
    if not text or len(text.strip()) < 50:
        return False
    alphanumeric = sum(1 for c in text if c.isalnum())
    return (alphanumeric / max(len(text), 1)) > 0.4

def extract_text_pdfplumber(pdf_path):
    page_texts, full_text = [], ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                try:
                    # x_tolerance=1.5 (default is 3) -- the default is too
                    # generous for some PDFs' font/kerning (confirmed on
                    # The Cambridge Ancient History vols) and merges
                    # adjacent words with no space between them, e.g.
                    # "defiantChristiansarenot...". Verified this tighter
                    # tolerance fixes that without over-splitting real
                    # words on multiple already-clean books.
                    text = page.extract_text(x_tolerance=1.5) or ""
                    page_texts.append((i + 1, text))
                    full_text += text + "\n\n"
                except Exception:
                    page_texts.append((i + 1, ""))
        return full_text, page_texts, total_pages
    except Exception:
        return "", [], 0

def extract_text_ocr(pdf_path, poppler_path=None, batch_size=10, max_render_height=2200):
    """
    OCR a PDF page-by-page in small batches instead of rasterizing the
    whole document in one convert_from_path() call. Large or poorly-
    scanned books at 300 DPI can otherwise buffer hundreds of full-page
    bitmaps in memory at once (this is what pdf2image does internally
    when asked to convert an entire document in one shot), which can
    raise a MemoryError deep inside its subprocess pipe-reading thread.
    Batching bounds peak memory regardless of document size.

    Some scanned PDFs (confirmed on a 1783 reprint in this corpus) have an
    oversized embedded page size, so "300 DPI" alone can render a single
    page as a ~6800x9750px image (~200MB raw) — large enough that even a
    10-page batch can exhaust memory or get OOM-killed before Python ever
    raises a catchable MemoryError. Passing `size=(None, max_render_height)`
    tells poppler to render directly at a capped resolution (not render-
    then-downscale), which bounds per-page memory regardless of the
    source page's declared dimensions. 2200px tall is still plenty for
    Tesseract to read clearly scanned book text.
    """
    print(f"      → Running OCR (may take several minutes for large files)...")
    kwargs = {"poppler_path": poppler_path} if poppler_path else {}
    size = (None, max_render_height)

    try:
        info = pdfinfo_from_path(pdf_path, **kwargs)
        total_pages = info.get("Pages", 0) or 0
    except Exception as e:
        print(f"      ⚠️  Could not read page count via pdfinfo: {e}")
        total_pages = 0

    if not total_pages:
        return "", [], 0

    page_texts, full_text = [], ""
    page_num = 1

    while page_num <= total_pages:
        last = min(page_num + batch_size - 1, total_pages)
        images = None
        try:
            images = convert_from_path(pdf_path, dpi=300, size=size,
                                        first_page=page_num, last_page=last, **kwargs)
        except MemoryError:
            print(f"\n      ⚠️  Out of memory rendering pages {page_num}-{last} "
                  f"as a batch — retrying one page at a time...")
        except Exception as e:
            print(f"\n      ⚠️  Render failed for pages {page_num}-{last}: {e}")

        if images is not None:
            for offset, image in enumerate(images):
                p = page_num + offset
                print(f"         OCR page {p}/{total_pages}...", end="\r")
                try:
                    text = pytesseract.image_to_string(image, lang="eng")
                except Exception as e:
                    print(f"\n         ⚠️  OCR failed on page {p}: {e}")
                    text = ""
                page_texts.append((p, text))
                full_text += text + "\n\n"
            del images
            gc.collect()
        else:
            # Fallback: render this batch one page at a time (much lower
            # peak memory, just slower).
            for p in range(page_num, last + 1):
                print(f"         OCR page {p}/{total_pages}...", end="\r")
                text = ""
                try:
                    single = convert_from_path(pdf_path, dpi=300, size=size,
                                                first_page=p, last_page=p, **kwargs)
                    if single:
                        text = pytesseract.image_to_string(single[0], lang="eng")
                    del single
                except Exception as e:
                    print(f"\n         ⚠️  OCR failed on page {p}: {e}")
                page_texts.append((p, text))
                full_text += text + "\n\n"
                gc.collect()

        page_num = last + 1

    print()
    return full_text, page_texts, total_pages

def extract_pdf(pdf_path):
    """Try pdfplumber first; fall back to OCR if text looks bad."""
    text, page_texts, total_pages = extract_text_pdfplumber(pdf_path)

    if total_pages == 0:
        print(f"      ⚠️  pdfplumber failed. Trying OCR...")
        text, page_texts, total_pages = extract_text_ocr(pdf_path, POPPLER_PATH)
        return text, page_texts, total_pages, True

    avg_chars = len(text) / max(total_pages, 1)
    if not text_looks_real(text) or avg_chars < OCR_THRESHOLD:
        print(f"      ⚠️  Poor text quality ({avg_chars:.0f} chars/page). Trying OCR...")
        ocr_text, ocr_pages, ocr_total = extract_text_ocr(pdf_path, POPPLER_PATH)
        if ocr_text and len(ocr_text) > len(text):
            return ocr_text, ocr_pages, ocr_total, True
    return text, page_texts, total_pages, False


def extract_txt(txt_path):
    """Read a plain text file. Returns (text, total_pages=0, ocr_used=False)."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return text, 0, False
    except Exception as e:
        print(f"      ⚠️  Could not read text file: {e}")
        return "", 0, False


# ─────────────────────────────────────────────────────────────
# Chunking (identical to the boxing archive)
# ─────────────────────────────────────────────────────────────

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]


# ─────────────────────────────────────────────────────────────
# Hand-verified metadata for the 60 curated files
# ─────────────────────────────────────────────────────────────
# Filled in via direct inspection of Final_Ingestion_List.md / Curation_List.md
# plus pdftotext checks on title pages for the academic articles and JSTOR
# pieces (DTIC_ADA471448, b21464625, 263434, 3286964, Bouchier). Where an
# author/year genuinely could not be confirmed, the field is left blank/0
# rather than guessed — better an honest gap than a false citation.
#
# Fields: title, author, year_published, volume_number, is_primary_source,
#         document_type, discipline, era, time_period,
#         civilization_or_nation, conflict_focus, geographic_focus,
#         subject_commander

FILE_METADATA = {

    # ---- Gibbon's Decline and Fall, 5-volume clean set ----
    "01.pdf": dict(title="The History of the Decline and Fall of the Roman Empire, Vol. I",
        author="Edward Gibbon", year_published=1776, volume_number=1, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Roman Empire decline through Byzantine fall, c. 180-1453 CE",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),
    "02.pdf": dict(title="The History of the Decline and Fall of the Roman Empire, Vol. II",
        author="Edward Gibbon", year_published=1776, volume_number=2, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Roman Empire decline through Byzantine fall, c. 180-1453 CE",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),
    "03.pdf": dict(title="The History of the Decline and Fall of the Roman Empire, Vol. III",
        author="Edward Gibbon", year_published=1776, volume_number=3, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Roman Empire decline through Byzantine fall, c. 180-1453 CE",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),
    "04.pdf": dict(title="The History of the Decline and Fall of the Roman Empire, Vol. IV",
        author="Edward Gibbon", year_published=1776, volume_number=4, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Roman Empire decline through Byzantine fall, c. 180-1453 CE",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),
    "05.pdf": dict(title="The History of the Decline and Fall of the Roman Empire, Vol. V",
        author="Edward Gibbon", year_published=1776, volume_number=5, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Roman Empire decline through Byzantine fall, c. 180-1453 CE",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),

    "AdamFerguson-TheHistoryOfTheProgressAndTerminationOfTheRomanRepublic-Complete1783.pdf": dict(
        title="The History of the Progress and Termination of the Roman Republic",
        author="Adam Ferguson", year_published=1783, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic",
        civilization_or_nation="roman", conflict_focus="roman_republic_wars",
        geographic_focus="mediterranean", subject_commander=""),

    "Eadie_1967_Roman_Mailed_Cavalry.pdf": dict(
        title="The Development of Roman Mailed Cavalry",
        author="John W. Eadie", year_published=1967, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="cavalry",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "general-military-the-roman-army-the-greatest-war-m..pdf": dict(
        title="The Roman Army: The Greatest War Machine",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="general", discipline="general_military_history",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "DTIC_ADA471448.pdf": dict(
        title="From Citizen Militia to Professional Military: The Transformation of the Roman Army",
        author="Robert Verlič", year_published=2007, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="administration_and_discipline",
        era="ancient", time_period="Roman Republic to Empire transition",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "263434.pdf": dict(
        title="Mutiny in the Roman Army: The Republic",
        author="William Stuart Messer", year_published=0, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="administration_and_discipline",
        era="ancient", time_period="Roman Republic",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "3286964.pdf": dict(
        title="Medicine in the Roman Army",
        author="Eugene Hugh Byrne", year_published=0, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="military_medicine",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "b21464625.pdf": dict(
        title="Was the Roman Army Provided with Medical Officers?",
        author="J. Y. Simpson", year_published=1856, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="military_medicine",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "bim_eighteenth-century_bibliotheca-topographic_1786.pdf": dict(
        title="Agricola's Campaign in Scotland",
        author="", year_published=1786, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="campaign_history",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_conquest_of_britain",
        geographic_focus="europe", subject_commander="Agricola"),

    "Nock_1952_Roman_army_religious_year.pdf": dict(
        title="The Roman Army and the Religious Year",
        author="Arthur Darby Nock", year_published=1952, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="religion_and_culture",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_army_organization",
        geographic_focus="mediterranean", subject_commander=""),

    "Magie_1923_Roman_Policy_Armenia.pdf": dict(
        title="Roman Policy in Armenia and Transcaucasia",
        author="David Magie", year_published=1923, volume_number=0, is_primary_source=False,
        document_type="academic_article", discipline="diplomacy_and_policy",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_eastern_frontier_diplomacy",
        geographic_focus="middle_east", subject_commander=""),

    "Bouchier_Syria_Roman_Province.pdf": dict(
        title="Syria as a Roman Province",
        author="Bouchier", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Empire",
        civilization_or_nation="roman", conflict_focus="roman_eastern_provinces",
        geographic_focus="middle_east", subject_commander=""),

    "N.Hooke-TheRomanHistoryFromTheBuildingOfRomeToTheRuinOfTheCommonwealth-Complete-4Vols.pdf": dict(
        title="The Roman History, from the Building of Rome to the Ruin of the Commonwealth",
        author="Nathaniel Hooke", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic",
        civilization_or_nation="roman", conflict_focus="roman_republic_wars",
        geographic_focus="mediterranean", subject_commander=""),

    "The illustrated history of Rome and the Roman empire (1877).pdf": dict(
        title="The Illustrated History of Rome and the Roman Empire",
        author="", year_published=1877, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic and Empire",
        civilization_or_nation="roman", conflict_focus="roman_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    "TheHistoryOfAncientRome.pdf": dict(
        title="The History of Ancient Rome",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic and Empire",
        civilization_or_nation="roman", conflict_focus="roman_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    "TheRomanAntiquitiesOfDionysiusHalicarnassensis1758-Complete4Vols..pdf": dict(
        title="The Roman Antiquities of Dionysius of Halicarnassus",
        author="Dionysius of Halicarnassus", year_published=1758, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Roman Kingdom and early Republic",
        civilization_or_nation="roman", conflict_focus="early_roman_history",
        geographic_focus="mediterranean", subject_commander=""),

    "sourcebookofroma0000munr.pdf": dict(
        title="A Source Book of Roman History",
        author="Dana Carleton Munro", year_published=1904, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Roman Republic and Empire",
        civilization_or_nation="roman", conflict_focus="roman_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    "p1historyofromer05duruuoft.pdf": dict(
        title="History of Rome, Part 1",
        author="Victor Duruy", year_published=0, volume_number=1, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic and Empire",
        civilization_or_nation="roman", conflict_focus="roman_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    "p2historyofromer01duruuoft.pdf": dict(
        title="History of Rome, Part 2",
        author="Victor Duruy", year_published=0, volume_number=2, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman Republic and Empire",
        civilization_or_nation="roman", conflict_focus="roman_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    # ---- Primary / classical military sources ----
    "TheTactiksOfAelian1616.pdf": dict(
        title="The Tactiks of Aelian",
        author="Aelian", year_published=1616, volume_number=0, is_primary_source=True,
        document_type="military_treatise", discipline="tactics_and_strategy",
        era="ancient", time_period="Roman Imperial period (Greek tactical tradition)",
        civilization_or_nation="greek", conflict_focus="greek_military_tactics",
        geographic_focus="mediterranean", subject_commander=""),

    "Thucydides-TheHistoryOfTheGrecianWarInEightBooks1676.pdf": dict(
        title="The History of the Grecian War",
        author="Thucydides", year_published=1676, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Classical Greece (Peloponnesian War)",
        civilization_or_nation="greek", conflict_focus="peloponnesian_war",
        geographic_focus="mediterranean", subject_commander=""),

    "Xenophon-TheInstitutionAndLifeOfCyrusTheGreat1685.pdf": dict(
        title="The Institution and Life of Cyrus the Great",
        author="Xenophon", year_published=1685, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="leadership_and_command",
        era="ancient", time_period="Achaemenid Persia",
        civilization_or_nation="persian", conflict_focus="persian_empire_formation",
        geographic_focus="middle_east", subject_commander="Cyrus the Great"),

    "Plutarch-TheLivesOfTheNobleGreciansAndRomans1676.pdf": dict(
        title="The Lives of the Noble Grecians and Romans",
        author="Plutarch", year_published=1676, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="leadership_and_command",
        era="ancient", time_period="Classical Greece and Rome",
        civilization_or_nation="roman, greek", conflict_focus="general",
        geographic_focus="mediterranean", subject_commander=""),

    "Tacitus_Histories_Loeb.pdf": dict(
        title="The Histories",
        author="Tacitus", year_published=0, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Roman Empire (69-96 CE)",
        civilization_or_nation="roman", conflict_focus="year_of_four_emperors",
        geographic_focus="mediterranean", subject_commander=""),

    "TheWholeGenuineAndCompleteWorksOfFlaviusJosephusTheJewishHistorian1792.pdf": dict(
        title="The Whole Genuine and Complete Works of Flavius Josephus",
        author="Flavius Josephus", year_published=1792, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Roman Empire (1st century CE)",
        civilization_or_nation="roman, jewish", conflict_focus="jewish_roman_wars",
        geographic_focus="middle_east", subject_commander=""),

    "Eadie_1967_Festus_Breviarium.pdf": dict(
        title="Festus' Breviarium (ed. J.W. Eadie)",
        author="Festus (ed. J.W. Eadie)", year_published=1967, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Late Roman Empire (4th century CE)",
        civilization_or_nation="roman", conflict_focus="roman_military_summary",
        geographic_focus="mediterranean", subject_commander=""),

    "Banchich-Lane_2009_ Zonaras_History.pdf": dict(
        title="The History of Zonaras (trans. Banchich & Lane)",
        author="Zonaras (trans. Banchich & Lane)", year_published=2009, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="byzantine", time_period="Byzantine compilation of Roman history (12th-century chronicle)",
        civilization_or_nation="byzantine, roman", conflict_focus="roman_byzantine_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    # ---- Empire-level conflict & decline ----
    "A History of the Later Roman Empire, AD 284-641.pdf": dict(
        title="A History of the Later Roman Empire, AD 284-641",
        author="Stephen Mitchell", year_published=2007, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Late Roman Empire to early Byzantine (284-641 CE)",
        civilization_or_nation="roman, byzantine", conflict_focus="roman_decline_and_fall",
        geographic_focus="mediterranean", subject_commander=""),

    "HerodiansHistoryOfHisOwnTimesOrOfTheRomanEmpireAfterMarcus.pdf": dict(
        title="Herodian's History of His Own Times, or, Of the Roman Empire After Marcus",
        author="Herodian", year_published=0, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="general_military_history",
        era="ancient", time_period="Roman Empire (180-238 CE)",
        civilization_or_nation="roman", conflict_focus="roman_imperial_crisis_3rd_century",
        geographic_focus="mediterranean", subject_commander=""),

    "TheLivesOfTheRomanEmperorsFromDomitianToConstantineTheGreat.pdf": dict(
        title="The Lives of the Roman Emperors from Domitian to Constantine the Great",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="leadership_and_command",
        era="ancient", time_period="Roman Empire (81-337 CE)",
        civilization_or_nation="roman", conflict_focus="roman_imperial_succession",
        geographic_focus="mediterranean", subject_commander=""),

    "Ten Caesars Roman Emperors from Augustus to Constantine pg226.pdf": dict(
        title="Ten Caesars: Roman Emperors from Augustus to Constantine",
        author="Barry Strauss", year_published=2019, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="leadership_and_command",
        era="ancient", time_period="Roman Empire (27 BCE-337 CE)",
        civilization_or_nation="roman", conflict_focus="roman_imperial_succession",
        geographic_focus="mediterranean", subject_commander=""),

    "The Cambridge Ancient History, Vol. 7, Part 2.pdf": dict(
        title="The Cambridge Ancient History, Vol. 7 Part 2: The Rise of Rome to 220 BC",
        author="", year_published=0, volume_number=7, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Rise of Rome (to 220 BCE)",
        civilization_or_nation="roman, greek", conflict_focus="rise_of_rome",
        geographic_focus="mediterranean", subject_commander=""),
    "The Cambridge Ancient History, Vol. 8.pdf": dict(
        title="The Cambridge Ancient History, Vol. 8: Rome and the Mediterranean to 133 BC",
        author="", year_published=0, volume_number=8, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman conquest of the Mediterranean (264-133 BCE)",
        civilization_or_nation="roman, greek, carthaginian", conflict_focus="punic_wars_and_mediterranean_conquest",
        geographic_focus="mediterranean", subject_commander=""),
    "The Cambridge Ancient History, Vol. 10.pdf": dict(
        title="The Cambridge Ancient History, Vol. 10: The Augustan Empire",
        author="", year_published=0, volume_number=10, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Augustan Empire (43 BCE-69 CE)",
        civilization_or_nation="roman", conflict_focus="augustan_empire",
        geographic_focus="mediterranean", subject_commander=""),
    "The Cambridge Ancient History, Vol. 11.pdf": dict(
        title="The Cambridge Ancient History, Vol. 11: The High Empire",
        author="", year_published=0, volume_number=11, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="High Roman Empire (70-192 CE)",
        civilization_or_nation="roman", conflict_focus="high_roman_empire",
        geographic_focus="mediterranean", subject_commander=""),
    "The Cambridge Ancient History, Vol. 12.pdf": dict(
        title="The Cambridge Ancient History, Vol. 12: The Crisis of Empire",
        author="", year_published=0, volume_number=12, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Crisis of the Roman Empire (193-337 CE)",
        civilization_or_nation="roman", conflict_focus="roman_imperial_crisis_3rd_century",
        geographic_focus="mediterranean", subject_commander=""),

    "Greece_under_the_Romans.pdf": dict(
        title="Greece Under the Romans",
        author="George Finlay", year_published=1844, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman rule of Greece (146 BCE-717 CE)",
        civilization_or_nation="greek, roman", conflict_focus="roman_conquest_of_greece",
        geographic_focus="mediterranean", subject_commander=""),

    "JohnGillies-TheHistoryOfAncientGreece-Complete1786.pdf": dict(
        title="The History of Ancient Greece",
        author="John Gillies", year_published=1786, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Classical Greece through Alexander",
        civilization_or_nation="greek", conflict_focus="greek_wars_general",
        geographic_focus="mediterranean", subject_commander=""),

    "TheHistoryOfGreeceFromAlexanderOfMacedonTillItsFinalSubjectionToTheRomanPower1782.pdf": dict(
        title="The History of Greece from Alexander of Macedon till its Final Subjection to the Roman Power",
        author="", year_published=1782, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Hellenistic Greece to Roman conquest (323-146 BCE)",
        civilization_or_nation="greek, roman", conflict_focus="hellenistic_to_roman_conquest",
        geographic_focus="mediterranean", subject_commander="Alexander the Great"),

    "TheCompleatHistoryOfTheAntientEgyptiansBabyloniansRomansAssyriansMedesPersiansGreciansAndCarthaginians.pdf": dict(
        title="The Compleat History of the Antient Egyptians, Babylonians, Romans, Assyrians, Medes, Persians, Grecians, and Carthaginians",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Ancient Near East and Mediterranean (multiple empires)",
        civilization_or_nation="egyptian, babylonian, roman, assyrian, persian, greek, carthaginian",
        conflict_focus="ancient_near_east_and_mediterranean_general",
        geographic_focus="mediterranean", subject_commander=""),

    "Hannibal Eng.pdf": dict(
        title="Hannibal",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="campaign_history",
        era="ancient", time_period="Second Punic War (218-201 BCE)",
        civilization_or_nation="carthaginian, roman", conflict_focus="punic_wars",
        geographic_focus="mediterranean", subject_commander="Hannibal"),

    # ---- Byzantine / Ottoman / Arab conflict history ----
    "Byzantium and the Arabs in the Fourth Century.pdf": dict(
        title="Byzantium and the Arabs in the Fourth Century",
        author="Irfan Shahid", year_published=1984, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="byzantine", time_period="Byzantine-Arab relations (4th century CE)",
        civilization_or_nation="byzantine, arab", conflict_focus="byzantine_arab_relations",
        geographic_focus="middle_east", subject_commander=""),

    "Shahid_Byzantium-and-the-Arabs-in-the-Fifth-Century_WEB.pdf": dict(
        title="Byzantium and the Arabs in the Fifth Century",
        author="Irfan Shahid", year_published=1989, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="byzantine", time_period="Byzantine-Arab relations (5th century CE)",
        civilization_or_nation="byzantine, arab", conflict_focus="byzantine_arab_relations",
        geographic_focus="middle_east", subject_commander=""),

    "Shahid_Byzantium-and-the-Arabs-in-the-Sixth-Century_V1P1_WEB.pdf": dict(
        title="Byzantium and the Arabs in the Sixth Century, Vol. 1 Part 1",
        author="Irfan Shahid", year_published=1995, volume_number=1, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="byzantine", time_period="Byzantine-Arab relations (6th century CE)",
        civilization_or_nation="byzantine, arab", conflict_focus="byzantine_arab_relations",
        geographic_focus="middle_east", subject_commander=""),

    "Shahid_Rome-and-the-Arabs_WEB.pdf": dict(
        title="Rome and the Arabs",
        author="Irfan Shahid", year_published=1984, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ancient", time_period="Roman-Arab relations (pre-4th century CE)",
        civilization_or_nation="roman, arab", conflict_focus="roman_arab_relations",
        geographic_focus="middle_east", subject_commander=""),

    "The Cambridge History of the Byzantine Empire c.500-1492 (2009).pdf": dict(
        title="The Cambridge History of the Byzantine Empire c.500-1492",
        author="", year_published=2009, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="byzantine", time_period="Byzantine Empire (500-1492 CE)",
        civilization_or_nation="byzantine", conflict_focus="byzantine_general_history",
        geographic_focus="mediterranean", subject_commander=""),

    "The Fall of Constantinople. The Ottoman Conquest of Byzantium.pdf": dict(
        title="The Fall of Constantinople: The Ottoman Conquest of Byzantium",
        author="", year_published=0, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="siege_warfare",
        era="ottoman", time_period="Fall of Constantinople (1453 CE)",
        civilization_or_nation="byzantine, ottoman", conflict_focus="fall_of_constantinople",
        geographic_focus="middle_east", subject_commander="Mehmed II"),

    "Cambridge History of Turkey - Volume 1.pdf": dict(
        title="The Cambridge History of Turkey, Vol. 1: Byzantium to Turkey, 1071-1453",
        author="", year_published=0, volume_number=1, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ottoman", time_period="Byzantium to Turkey (1071-1453 CE)",
        civilization_or_nation="ottoman, byzantine, turkish", conflict_focus="ottoman_general_history",
        geographic_focus="middle_east", subject_commander=""),
    "Cambridge History of Turkey - Volume 2.pdf": dict(
        title="The Cambridge History of Turkey, Vol. 2: The Ottoman Empire as a World Power, 1453-1603",
        author="", year_published=0, volume_number=2, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ottoman", time_period="Ottoman Empire as world power (1453-1603 CE)",
        civilization_or_nation="ottoman, turkish", conflict_focus="ottoman_general_history",
        geographic_focus="middle_east", subject_commander=""),
    "Cambridge History of Turkey - Volume 3.pdf": dict(
        title="The Cambridge History of Turkey, Vol. 3: The Later Ottoman Empire, 1603-1839",
        author="", year_published=0, volume_number=3, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ottoman", time_period="Later Ottoman Empire (1603-1839 CE)",
        civilization_or_nation="ottoman, turkish", conflict_focus="ottoman_general_history",
        geographic_focus="middle_east", subject_commander=""),
    "Cambridge History of Turkey - Volume 4.pdf": dict(
        title="The Cambridge History of Turkey, Vol. 4: Turkey in the Modern World",
        author="", year_published=0, volume_number=4, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="modern", time_period="Turkey in the modern world (1839 CE-present)",
        civilization_or_nation="ottoman, turkish", conflict_focus="ottoman_general_history",
        geographic_focus="middle_east", subject_commander=""),

    "TheHistoryOfTheGrowthAndDecayOfTheOthmanEmpire1734.pdf": dict(
        title="The History of the Growth and Decay of the Othman Empire",
        author="Demetrius Cantemir", year_published=1734, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="ottoman", time_period="Ottoman Empire (origins to early 18th century)",
        civilization_or_nation="ottoman", conflict_focus="ottoman_general_history",
        geographic_focus="middle_east", subject_commander=""),

    "The Fall of the Ottomans- The Great War in the Middle East.pdf": dict(
        title="The Fall of the Ottomans: The Great War in the Middle East",
        author="Eugene Rogan", year_published=2015, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="modern", time_period="World War I (1914-1918 CE)",
        civilization_or_nation="ottoman", conflict_focus="world_war_one_middle_east",
        geographic_focus="middle_east", subject_commander=""),

    "History of the Arabs_ From the Earliest Times to the Present.pdf": dict(
        title="History of the Arabs: From the Earliest Times to the Present",
        author="Philip K. Hitti", year_published=1937, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="general_military_history",
        era="multi_era", time_period="Pre-Islamic Arabia through the modern era",
        civilization_or_nation="arab", conflict_focus="arab_general_history",
        geographic_focus="middle_east", subject_commander=""),

    # ---- Other historical conquests ----
    "TheDiscoveryAndConquestOfTheProvincesOfPeru1581.pdf": dict(
        title="The Discovery and Conquest of the Provinces of Peru",
        author="", year_published=1581, volume_number=0, is_primary_source=True,
        document_type="primary_source_text", discipline="campaign_history",
        era="early_modern_conquest", time_period="Spanish conquest of the Inca Empire (1532-1572 CE)",
        civilization_or_nation="spanish, inca", conflict_focus="spanish_conquest_of_peru",
        geographic_focus="americas", subject_commander="Francisco Pizarro"),

    "TheHistoryOfTheConquestOfMexicoByTheSpaniards1724.pdf": dict(
        title="The History of the Conquest of Mexico by the Spaniards",
        author="Antonio de Solis", year_published=1724, volume_number=0, is_primary_source=False,
        document_type="historical_chronicle", discipline="campaign_history",
        era="early_modern_conquest", time_period="Spanish conquest of the Aztec Empire (1519-1521 CE)",
        civilization_or_nation="spanish, aztec", conflict_focus="spanish_conquest_of_mexico",
        geographic_focus="americas", subject_commander="Hernan Cortes"),
}


# ─────────────────────────────────────────────────────────────
# Fallback classifiers — only used for a file NOT in FILE_METADATA
# (e.g. something added to the manifest later that hasn't been
# hand-verified yet). Mirrors the boxing archive's filename-driven
# approach so the pipeline never hard-fails on an unknown file.
# ─────────────────────────────────────────────────────────────

def filename_to_title(filename):
    name = Path(filename).stem
    name = re.sub(r'[_\-\.]+', ' ', name)
    return name.title()


def extract_author_fallback(filename):
    stem = Path(filename).stem
    if ' - ' in stem:
        candidate = stem.split(' - ')[0].strip()
        if len(candidate) < 50 and not re.search(r'\b\d{4}\b', candidate):
            return candidate
    return ''


def extract_year_fallback(filename):
    matches = re.findall(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', filename)
    return int(matches[0]) if matches else 0


KNOWN_COMMANDERS = [
    'Hannibal', 'Caesar', 'Agricola', 'Constantine', 'Augustus', 'Trajan',
    'Hadrian', 'Alexander', 'Cyrus', 'Cortes', 'Pizarro', 'Mehmed',
    'Scipio', 'Pompey', 'Sulla', 'Marius', 'Belisarius',
]

CIVILIZATION_KEYWORDS = {
    'roman': ['roman', 'rome'],
    'greek': ['greek', 'greece', 'grecian'],
    'byzantine': ['byzantine', 'byzantium', 'constantinople'],
    'ottoman': ['ottoman', 'othman', 'turkey', 'turkish'],
    'arab': ['arab'],
    'persian': ['persia', 'cyrus'],
    'egyptian': ['egyptian'],
    'carthaginian': ['carthag', 'hannibal'],
    'spanish': ['spaniard', 'spanish', 'spain'],
    'inca': ['peru', 'inca'],
    'aztec': ['mexico', 'aztec'],
}

GEOGRAPHY_MAP = {
    'roman': 'mediterranean', 'greek': 'mediterranean', 'byzantine': 'mediterranean',
    'carthaginian': 'mediterranean', 'persian': 'middle_east', 'egyptian': 'mediterranean',
    'arab': 'middle_east', 'ottoman': 'middle_east', 'spanish': 'europe',
    'inca': 'americas', 'aztec': 'americas',
}


def classify_document_type_fallback(filename):
    n = filename.lower()
    if any(k in n for k in ['cambridge', 'history of', 'chronicle', 'empire']):
        return 'historical_chronicle'
    if any(k in n for k in ['biography', 'life of']):
        return 'biography'
    if any(k in n for k in ['tactics', 'tactiks', 'treatise', 'manual']):
        return 'military_treatise'
    return 'general'


def classify_discipline_fallback(filename):
    n = filename.lower()
    if 'cavalry' in n: return 'cavalry'
    if 'medic' in n: return 'military_medicine'
    if 'religio' in n: return 'religion_and_culture'
    if any(k in n for k in ['policy', 'diplomac']): return 'diplomacy_and_policy'
    if any(k in n for k in ['mutiny', 'discipline', 'militia']): return 'administration_and_discipline'
    if any(k in n for k in ['campaign', 'conquest']): return 'campaign_history'
    if any(k in n for k in ['tactics', 'tactiks', 'strategy']): return 'tactics_and_strategy'
    if 'siege' in n: return 'siege_warfare'
    return 'general_military_history'


def classify_civilization_fallback(filename):
    n = filename.lower()
    found = [civ for civ, keys in CIVILIZATION_KEYWORDS.items() if any(k in n for k in keys)]
    return ', '.join(found) if found else ''


def classify_geography_fallback(civilization_str):
    for civ in civilization_str.split(', '):
        if civ in GEOGRAPHY_MAP:
            return GEOGRAPHY_MAP[civ]
    return 'unknown'


def extract_commander_fallback(filename):
    found = [c for c in KNOWN_COMMANDERS if c.lower() in filename.lower()]
    return ', '.join(found) if found else ''


def build_metadata(filename):
    """
    Look up the hand-verified table first (the 60 curated files). Anything
    not found there (a future addition to the manifest) gets generic
    filename-keyword classification instead, same fallback spirit as the
    boxing archive.
    """
    if filename in FILE_METADATA:
        meta = dict(FILE_METADATA[filename])
    else:
        print(f"      ⚠️  '{filename}' not in hand-verified table — using fallback classifiers.")
        year = extract_year_fallback(filename)
        civilization = classify_civilization_fallback(filename)
        meta = dict(
            title=filename_to_title(filename),
            author=extract_author_fallback(filename),
            year_published=year,
            volume_number=0,
            is_primary_source=False,
            document_type=classify_document_type_fallback(filename),
            discipline=classify_discipline_fallback(filename),
            era='unknown',
            time_period='unknown',
            civilization_or_nation=civilization,
            conflict_focus='general',
            geographic_focus=classify_geography_fallback(civilization),
            subject_commander=extract_commander_fallback(filename),
        )

    # Phase 2 fields — blank for now, filled by a later script
    meta.update({
        "commanders_mentioned":         '',
        "battles_mentioned":            '',
        "units_mentioned":              '',
        "weapons_or_tech_mentioned":    '',
        "contains_battle_account":      False,
        "contains_tactical_analysis":   False,
        "contains_strategic_overview":  False,
        "contains_biographical_info":   False,
        "has_statistics":               False,
    })
    return meta


# ─────────────────────────────────────────────────────────────
# Weaviate upload
# ─────────────────────────────────────────────────────────────

def upload_chunks(collection, chunks, source_file, metadata, total_pages, ocr_used):
    """Batch-upload chunks to Weaviate. Returns count of successful uploads."""
    uploaded = 0
    with collection.batch.dynamic() as batch:
        for i, chunk in enumerate(chunks):
            try:
                batch.add_object(properties={
                    "content":    chunk,
                    "source_file": source_file,
                    "page_number": 0,
                    "chunk_index": i,
                    "total_pages": total_pages,
                    "ocr_used":    ocr_used,
                    **metadata,
                })
                uploaded += 1
            except Exception as e:
                print(f"      ⚠️  Failed to upload chunk {i}: {e}")
    return uploaded


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Clear progress log and reprocess all manifest files")
    args = parser.parse_args()

    # Config checks
    if not WEAVIATE_URL or "YOUR-CLUSTER" in WEAVIATE_URL:
        print("❌ WEAVIATE_URL not set in .env file.")
        sys.exit(1)
    if not OPENAI_API_KEY or "your-openai" in OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in .env file.")
        sys.exit(1)
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Warfare PDF folder not found: {PDF_FOLDER}")
        sys.exit(1)

    # Tesseract
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    else:
        print(f"⚠️  Tesseract not found at {TESSERACT_PATH} — OCR won't work.\n")

    # Progress log
    if args.reset and os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print("🔄 Progress log reset.\n")
    log = load_log()

    # Manifest-filtered file discovery (NOT a full-folder glob — this
    # folder has 187 PDFs, only the curated subset below gets ingested)
    manifest_names = load_manifest(MANIFEST_FILE)
    all_pdfs_in_folder = {p.name: p for p in Path(PDF_FOLDER).glob("*.pdf")}
    all_pdfs_in_folder.update({p.name: p for p in Path(PDF_FOLDER).glob("*.PDF")})

    found_files = []
    missing_from_disk = []
    for name in sorted(manifest_names):
        if name in all_pdfs_in_folder:
            found_files.append(all_pdfs_in_folder[name])
        else:
            missing_from_disk.append(name)

    if missing_from_disk:
        print("⚠️  These manifest entries were not found on disk (skipping):")
        for m in missing_from_disk:
            print(f"   - {m}")
        print()

    already_done = set(log["completed"])
    to_process = [f for f in found_files if f.name not in already_done]

    print(f"\n📜 Historical Warfare Archive Ingestion")
    print(f"   Source folder: {PDF_FOLDER}")
    print(f"   Manifest file: {MANIFEST_FILE}")
    print(f"   Files in manifest:  {len(manifest_names)}")
    print(f"   Found on disk:      {len(found_files)}")
    print(f"   Already done:       {len(already_done)}")
    print(f"   To process:         {len(to_process)}")
    print(f"   Chunk size:         {CHUNK_SIZE} chars  (overlap: {CHUNK_OVERLAP})")

    if not to_process:
        print("\n✅ All manifest files already ingested! Use --reset to reprocess.")
        sys.exit(0)

    # Connect
    print(f"\n🔌 Connecting to Weaviate Cloud...")
    try:
        client     = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={"X-OpenAI-Api-Key": OPENAI_API_KEY},
        )
        collection = client.collections.get(COLLECTION_NAME)
        print("✅ Connected!\n")
    except Exception as e:
        print(f"❌ Could not connect: {e}")
        sys.exit(1)

    success_count = fail_count = total_chunks = 0

    for idx, file_path in enumerate(to_process, 1):
        filename = file_path.name
        print(f"[{idx}/{len(to_process)}] {filename}")

        try:
            # 1. Extract text
            text, page_texts, total_pages, ocr_used = extract_pdf(str(file_path))

            if not text or len(text.strip()) < 100:
                print(f"      ⚠️  Too little text extracted — skipping.")
                log["failed"][filename] = "Too little text"
                save_log(log)
                fail_count += 1
                continue

            text = clean_text(text)
            pages_display = f"{total_pages} pages" if total_pages else "unknown pages"
            print(f"      ✓ {len(text):,} chars | {pages_display}"
                  + (" [OCR]" if ocr_used else ""))

            # 2. Chunk
            chunks = chunk_text(text)
            print(f"      ✓ {len(chunks)} chunks")
            if not chunks:
                log["failed"][filename] = "No chunks"
                save_log(log)
                fail_count += 1
                continue

            # 3. Build metadata
            meta = build_metadata(filename)
            print(f"      ✓ {meta['document_type']} | {meta['discipline']} | "
                  f"{meta['era']} | {meta['civilization_or_nation']}"
                  + (f" | author: {meta['author']}" if meta['author'] else "")
                  + (f" | commander: {meta['subject_commander']}" if meta['subject_commander'] else "")
                  + (f" | vol {meta['volume_number']}" if meta['volume_number'] else "")
                  + (f" | {meta['year_published']}" if meta['year_published'] else ""))

            # 4. Upload
            uploaded = upload_chunks(collection, chunks, filename, meta,
                                     total_pages, ocr_used)
            print(f"      ✓ Uploaded {uploaded} chunks")

            total_chunks += uploaded
            log["completed"].append(filename)
            save_log(log)
            success_count += 1
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n⏸️  Stopped. Progress saved — run again to continue.")
            break
        except Exception as e:
            print(f"      ❌ Error: {e}")
            log["failed"][filename] = str(e)
            save_log(log)
            fail_count += 1

    client.close()
    print(f"\n{'='*55}")
    print(f"✅ Done!")
    print(f"   Successful:      {success_count} files")
    print(f"   Failed:          {fail_count} files")
    print(f"   Chunks uploaded: {total_chunks:,}")
    print(f"   Total in archive: {len(log['completed'])} files")

    if log["failed"]:
        print(f"\n⚠️  Failed files:")
        for fname, reason in log["failed"].items():
            print(f"   - {fname}: {reason}")

    print(f"\n👉 Next: search-app integration (querying this collection from the")
    print(f"   existing archive app) is the next phase once ingestion completes.")


if __name__ == "__main__":
    main()
