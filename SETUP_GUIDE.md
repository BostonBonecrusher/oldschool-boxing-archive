# Boxing Archive — Complete Setup Guide
### Read this top to bottom before running anything. Go one step at a time.

---

## What You're Building

A searchable database of ~300 boxing PDFs. You'll be able to type a question like *"Who trained Sugar Ray Robinson?"* and get an accurate, sourced answer pulled directly from your books.

The stack:
- **Python** — the language all our scripts are written in
- **Weaviate Cloud** — the database that stores and searches your content
- **OpenAI** — creates the "meaning fingerprints" (embeddings) that power smart search
- **pdfplumber + Tesseract** — reads your PDFs, including scanned ones that need OCR

---

## STEP 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Download the latest **Python 3.11** or **3.12** (not 3.13 yet — some libraries lag behind)
3. Run the installer
4. **CRITICAL:** On the first screen, check the box that says **"Add Python to PATH"** before clicking Install
5. When done, open **Command Prompt** (press Windows key, type `cmd`, hit Enter)
6. Type this and hit Enter:
   ```
   python --version
   ```
   You should see something like `Python 3.11.9`. If you see an error, Python didn't install correctly — go back and redo step 4.

---

## STEP 2 — Install Tesseract (for OCR on scanned PDFs)

Some of your PDFs are scanned images, not real text. Tesseract reads those.

1. Go to: **https://github.com/UB-Mannheim/tesseract/wiki**
2. Download the Windows installer (the `.exe` file)
3. Run it — keep all defaults, just click Next through everything
4. **Important:** Note where it installs. Default is usually:
   ```
   C:\Program Files\Tesseract-OCR\tesseract.exe
   ```
5. To verify, open Command Prompt and type:
   ```
   "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
   ```
   You should see a version number. If not, find where it actually installed and note that path.

---

## STEP 3 — Install Poppler (needed for PDF-to-image conversion for OCR)

1. Go to: **https://github.com/oschwartz10612/poppler-windows/releases/**
2. Download the latest `.zip` file (e.g., `Release-24.xx.x-0.zip`)
3. Extract it somewhere permanent, like `C:\poppler\`
4. Inside you'll find a `bin` folder. The full path would be something like:
   ```
   C:\poppler\poppler-24.xx.x\Library\bin
   ```
   Note this path — you'll put it in your `.env` file later.

---

## STEP 4 — Set Up the Project Folder

Your project folder is:
```
C:\Users\tatte\OneDrive\Documents\Claude\Projects\Oldschool boxing Archive\
```

This is where all the scripts live. Your PDFs stay where they are:
```
C:\Users\tatte\OneDrive\Boxing_books\
```

---

## STEP 5 — Get Your API Keys

### Weaviate Cloud
1. Log into your Weaviate Cloud account at **https://console.weaviate.cloud/**
2. Click on your cluster
3. Find and copy two things:
   - **Cluster URL** — looks like `https://your-cluster-name.weaviate.network`
   - **API Key** — under "API Keys" section, copy the Admin key

### OpenAI
1. Go to **https://platform.openai.com/api-keys**
2. Click "Create new secret key"
3. Copy it immediately — you can't see it again after closing that window
4. Make sure you have credits loaded (go to Billing — $5 is plenty for all 300 PDFs)

---

## STEP 6 — Create Your .env File

In your project folder, there is a file called `.env.template`.

1. Make a copy of it
2. Rename the copy to `.env` (no .template, just `.env`)
3. Open it with Notepad or VS Code
4. Fill in all the values (instructions are inside the file)

**Never share your .env file or upload it anywhere.** It contains secret keys.

---

## STEP 7 — Install Python Libraries

1. Open Command Prompt
2. Type this command exactly and hit Enter:
   ```
   pip install weaviate-client openai pdfplumber pdf2image pytesseract python-dotenv tiktoken langchain-text-splitters pillow
   ```
3. Wait for it to finish (may take a couple minutes)
4. If you see any red error messages, copy them and share with Claude

---

## STEP 8 — Enable OpenAI Vectorizer in Weaviate Cloud

Before ingesting, you need to tell Weaviate Cloud to use OpenAI for embeddings:

1. Log into **https://console.weaviate.cloud/**
2. Go to your cluster settings
3. Look for **"Modules"** or **"Integrations"**
4. Enable **text2vec-openai**
5. You'll be asked to enter your OpenAI API key there too

---

## STEP 9 — Run the Scripts in Order

Once everything above is done, run scripts in this order:

**First — create the database structure:**
```
python 01_create_schema.py
```
Expected output: `✅ Collection 'BoxingChunk' created successfully.`

**Second — ingest your PDFs (this takes a while):**
```
python 02_ingest.py
```
This will process all 300 PDFs. Expect it to run for 30–90 minutes depending on how many need OCR. It shows progress as it goes and saves its place, so if it crashes you can just run it again and it picks up where it left off.

**Third — test your search:**
```
python 03_search.py
```
Type any boxing question and hit Enter.

---

## Troubleshooting

**"ModuleNotFoundError: No module named X"**
→ Run `pip install X` in Command Prompt

**"tesseract is not installed or it's not in your PATH"**
→ Check that Tesseract installed at `C:\Program Files\Tesseract-OCR\` and verify the path in your `.env` file

**"AuthenticationFailedError" from Weaviate**
→ Double-check your WEAVIATE_URL and WEAVIATE_API_KEY in your `.env` file. Make sure there are no extra spaces.

**Weaviate connection times out**
→ Check that your cluster is running in Weaviate Cloud console. Free-tier clusters sleep after inactivity.

**A PDF produces garbage text**
→ The script will auto-detect this and use OCR instead. If it still looks wrong, note the filename and tell Claude.

---

## File Map

```
Oldschool boxing Archive\
├── SETUP_GUIDE.md          ← You are here
├── .env.template           ← Copy this to .env and fill it in
├── .env                    ← YOUR secrets (never share this)
├── requirements.txt        ← List of Python libraries needed
├── 01_create_schema.py     ← Run once to set up database
├── 02_ingest.py            ← Run to process all PDFs
├── 03_search.py            ← Run to search your archive
└── ingestion_log.json      ← Created automatically, tracks progress
```
