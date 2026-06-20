"""
02_ingest.py
------------
Main ingestion script. For each PDF or TXT file it:
  1. Extracts text (auto-detects if OCR is needed for PDFs)
  2. Cleans and chunks the text
  3. Auto-tags every chunk with historian metadata
  4. Uploads to Weaviate

Saves progress as it goes — if it stops, run it again and it picks up
where it left off.

Usage:
    python 02_ingest.py

Re-process everything from scratch:
    python 02_ingest.py --reset
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

WEAVIATE_URL     = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PDF_FOLDER       = os.getenv("PDF_FOLDER", r"C:\Users\tatte\OneDrive\Boxing_books")
TESSERACT_PATH   = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
POPPLER_PATH     = os.getenv("POPPLER_PATH", None)
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))

COLLECTION_NAME = "BoxingChunk"
LOG_FILE        = "ingestion_log.json"
OCR_THRESHOLD   = 100  # avg chars/page below this triggers OCR


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
# PDF extraction
# ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x80-\xFF]', ' ', text)
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
                    text = page.extract_text() or ""
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

    Some scanned PDFs have an oversized embedded page size, so "300 DPI"
    alone can render a single page as a huge multi-hundred-MB image —
    large enough that even a small batch can exhaust memory or get
    OOM-killed before Python ever raises a catchable MemoryError. Passing
    `size=(None, max_render_height)` tells poppler to render directly at
    a capped resolution (not render-then-downscale), which bounds
    per-page memory regardless of the source page's declared dimensions.
    2200px tall is still plenty for Tesseract to read clearly scanned
    book text.
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
# Chunking
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
# Metadata extraction — all auto-derived from filename
# ─────────────────────────────────────────────────────────────

def filename_to_title(filename):
    """'Boxiana_vol3_1821.pdf'  →  'Boxiana Vol3 1821'"""
    name = Path(filename).stem
    name = re.sub(r'[_\-\.]+', ' ', name)
    return name.title()


def extract_author(filename):
    """
    Many files follow the pattern 'Author Name - Title.pdf'.
    Extract the author from before the first ' - '.
    Returns empty string if not found.
    """
    stem = Path(filename).stem
    if ' - ' in stem:
        candidate = stem.split(' - ')[0].strip()
        # Sanity: author shouldn't contain a 4-digit year or be very long
        if len(candidate) < 50 and not re.search(r'\b\d{4}\b', candidate):
            return candidate
    return ''


def extract_year(filename):
    """Pull first 4-digit year (1600–2029) from filename. Returns 0 if none."""
    matches = re.findall(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', filename)
    return int(matches[0]) if matches else 0


def year_to_decade(year):
    if year == 0:
        return 'unknown'
    return f"{(year // 10) * 10}s"


def year_to_era(year):
    if year == 0:       return 'unknown'
    if year < 1900:     return 'bare_knuckle_era'
    if year < 1950:     return 'golden_age'
    if year < 1980:     return 'midcentury'
    return 'modern'


def year_to_rules_era(year):
    """
    Boxing rules eras:
      London Prize Ring Rules  — pre-1867
      Queensberry Transition   — 1867–1900
      Modern Queensberry       — 1900+
    """
    if year == 0:       return 'unknown'
    if year < 1867:     return 'london_prize_ring'
    if year < 1901:     return 'queensberry_transition'
    return 'modern_queensberry'


def extract_volume_number(filename):
    """'Boxiana Vol 4 1828.pdf' → 4. Returns 0 if not found."""
    match = re.search(r'[Vv]ol(?:ume)?\.?\s*(\d+)', filename)
    return int(match.group(1)) if match else 0


def classify_document_type(filename):
    """
    Classify the type of document.
    Returns: historical_chronicle | training_manual | fight_account |
             biography | combat_manual | martial_arts | periodical | general
    """
    n = filename.lower()
    if any(k in n for k in ['boxiana', 'pancratia', 'annals', 'chronicles', 'history of']):
        return 'historical_chronicle'
    if any(k in n for k in [' vs ', '-vs-', 'fitzvs', 'pearson-fitz',
                              'life and battles', 'battles of']):
        return 'fight_account'
    if any(k in n for k in ['biography', ' bio ', 'life of', 'story of', 'memoirs']):
        return 'biography'
    if any(k in n for k in ['how to', 'training', 'lessons', 'guide', 'manual',
                              'technique', 'footwork', 'infighting', 'shadow boxing',
                              'art of', 'system of', 'science of', 'knock em']):
        return 'training_manual'
    if any(k in n for k in ['combat', 'gestapo', 'ss manual', 'amphibious',
                              'mercenary', 'self-defense', 'self defense', 'defence',
                              'hand to hand', 'handtohand', 'soldier']):
        return 'combat_manual'
    if any(k in n for k in ['shaolin', 'mma', 'kickboxing', 'martial arts',
                              'kung fu', 'karate']):
        return 'martial_arts'
    if any(k in n for k in ['volume', 'issue', 'vol ', 'vol_', 'magazine',
                              'journal', 'shedet', 'pages ']):
        return 'periodical'
    return 'general'


def classify_discipline(filename):
    """
    Returns: boxing | bare_knuckle | martial_arts | self_defense | mixed
    """
    n = filename.lower()
    if any(k in n for k in ['bare-knuckle', 'bare knuckle', 'bareknuckle',
                              'boxiana', 'pancratia', 'pugilism', 'prize fight',
                              'sullivan', 'yankee']):
        return 'bare_knuckle'
    if any(k in n for k in ['shaolin', 'mma', 'kickboxing', 'martial arts',
                              'kung fu', 'karate']):
        return 'martial_arts'
    if any(k in n for k in ['combat', 'gestapo', 'amphibious', 'mercenary',
                              'self-defense', 'self defense', 'defence',
                              'hand to hand', 'handtohand', 'soldier']):
        return 'self_defense'
    return 'boxing'


def classify_geographic_focus(filename):
    """Returns: uk | usa | international | unknown"""
    n = filename.lower()
    uk_hits  = sum(1 for k in ['boxiana', 'pancratia', 'british', 'england',
                                 'london', 'fairbairn', 'mendoza', 'jem '] if k in n)
    usa_hits = sum(1 for k in ['yankee', 'american', 'dempsey', 'sullivan',
                                 'new york', 'chicago', 'philadelphia'] if k in n)
    if uk_hits > usa_hits and uk_hits > 0:   return 'uk'
    if usa_hits > uk_hits and usa_hits > 0:  return 'usa'
    if any(k in n for k in ['international', 'world']):  return 'international'
    return 'unknown'


# Known fighters to look for in filenames
KNOWN_FIGHTERS = [
    'Sullivan', 'Dempsey', 'Fitzsimmons', 'Corbett', 'Robinson', 'Louis',
    'Armstrong', 'Greb', 'Walker', 'Tunney', 'Johnson', 'Jeffries', 'Sharkey',
    'Baer', 'Braddock', 'Schmeling', 'Marciano', 'Charles', 'Walcott',
    'Patterson', 'Liston', 'Ali', 'Frazier', 'Foreman', 'Norton', 'Duran',
    'Leonard', 'Hagler', 'Hearns', 'Chavez', 'Tyson', 'Holyfield', 'Miller',
    'Mendoza', 'Cribb', 'Pearson', 'Kaneko',
]

def extract_subject_fighter(filename):
    """
    Primary fighter(s) this document is about, based on filename.
    Returns comma-separated string, or empty string.
    """
    found = [f for f in KNOWN_FIGHTERS if f.lower() in filename.lower()]
    return ', '.join(found) if found else ''


def extract_fighters_in_title(filename):
    """
    For fight accounts: try to pull both fighters from a 'AxVsB' pattern.
    Falls back to subject_fighter.
    """
    stem = Path(filename).stem
    match = re.search(r'(\w+)[Vv][Ss]\.?(\w+)', stem)
    if match:
        return f"{match.group(1)}, {match.group(2)}"
    return extract_subject_fighter(filename)


def is_primary_source(year, doc_type):
    """True if the document was likely written at the time of the events."""
    if year == 0:
        return False
    if year < 1960 and doc_type in ('historical_chronicle', 'biography',
                                     'fight_account', 'periodical'):
        return True
    if year < 1940 and doc_type == 'training_manual':
        return True
    return False


def build_metadata(filename):
    """
    Run all classifiers and return a dict of every Phase 1 metadata field.
    Call this once per PDF before uploading chunks.
    """
    year     = extract_year(filename)
    doc_type = classify_document_type(filename)

    return {
        "title":             filename_to_title(filename),
        "author":            extract_author(filename),
        "year_published":    year,
        "decade_focus":      year_to_decade(year),
        "volume_number":     extract_volume_number(filename),
        "is_primary_source": is_primary_source(year, doc_type),
        "document_type":     doc_type,
        "discipline":        classify_discipline(filename),
        "rules_era":         year_to_rules_era(year),
        "era":               year_to_era(year),
        "subject_fighter":   extract_subject_fighter(filename),
        "fighters_in_title": extract_fighters_in_title(filename),
        "geographic_focus":  classify_geographic_focus(filename),
        # Phase 2 fields — blank for now, filled by a later script
        "weight_class_focus":         '',
        "fighters_mentioned":         '',
        "venues_mentioned":           '',
        "contains_training_methods":  False,
        "contains_fight_account":     False,
        "contains_biographical_info": False,
        "has_statistics":             False,
    }


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
                        help="Clear progress log and reprocess all PDFs")
    args = parser.parse_args()

    # Config checks
    if not WEAVIATE_URL or "YOUR-CLUSTER" in WEAVIATE_URL:
        print("❌ WEAVIATE_URL not set in .env file.")
        sys.exit(1)
    if not OPENAI_API_KEY or "your-openai" in OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not set in .env file.")
        sys.exit(1)
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ PDF folder not found: {PDF_FOLDER}")
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

    # Find PDFs and TXT files
    pdf_files  = sorted(Path(PDF_FOLDER).glob("**/*.pdf"))
    txt_files  = sorted(Path(PDF_FOLDER).glob("**/*.txt"))
    all_files  = pdf_files + txt_files
    already_done = set(log["completed"])
    to_process = [f for f in all_files if f.name not in already_done]

    print(f"\n📚 Boxing Archive Ingestion")
    print(f"   Source folder: {PDF_FOLDER}")
    print(f"   Total PDFs:    {len(pdf_files)}")
    print(f"   Total TXTs:    {len(txt_files)}")
    print(f"   Already done:  {len(already_done)}")
    print(f"   To process:    {len(to_process)}")
    print(f"   Chunk size:    {CHUNK_SIZE} chars  (overlap: {CHUNK_OVERLAP})")

    if not to_process:
        print("\n✅ All files already ingested! Use --reset to reprocess.")
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
        is_txt   = file_path.suffix.lower() == ".txt"
        print(f"[{idx}/{len(to_process)}] {filename}")

        try:
            # 1. Extract text
            if is_txt:
                text, total_pages, ocr_used = extract_txt(str(file_path))
                page_texts = []
            else:
                text, page_texts, total_pages, ocr_used = extract_pdf(str(file_path))

            if not text or len(text.strip()) < 100:
                print(f"      ⚠️  Too little text extracted — skipping.")
                log["failed"][filename] = "Too little text"
                save_log(log)
                fail_count += 1
                continue

            text = clean_text(text)
            pages_display = f"{total_pages} pages" if total_pages else "TXT"
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
                  f"{meta['rules_era']} | {meta['era']}"
                  + (f" | author: {meta['author']}" if meta['author'] else "")
                  + (f" | fighter: {meta['subject_fighter']}" if meta['subject_fighter'] else "")
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
    print(f"   Total in archive: {len(log['completed'])} PDFs")

    if log["failed"]:
        print(f"\n⚠️  Failed files:")
        for fname, reason in log["failed"].items():
            print(f"   - {fname}: {reason}")

    print(f"\n👉 Next: run  python 03_search.py")
    print(f"   Then later: run  python 04_tag_phase2.py  (AI metadata enrichment)")


if __name__ == "__main__":
    main()
