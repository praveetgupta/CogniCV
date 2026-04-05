# 🧠 CogniCV — Smart Resume Analyzer with Job Matching

A Streamlit web app that analyzes your resume against a job description and provides a match score, identifies missing skills, and offers improvement suggestions — all powered by NLP.

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### 2. Install Dependencies

```bash
pip install streamlit spacy scikit-learn PyPDF2 nltk
```

### 3. Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the App

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

### 5. Use It

1. **Upload** your resume (PDF or TXT).
2. **Paste** the target job description.
3. Click **Analyze Match**.
4. Review your match score, matched/missing skills, and improvement suggestions.

---

## How the NLP Pipeline Works

The analysis happens in five stages:

### Stage 1 — Text Extraction

When you upload a PDF, `PyPDF2` reads every page and concatenates the raw text. For TXT files, the bytes are decoded directly. This gives us the unstructured resume text to work with.

### Stage 2 — Preprocessing

Both the resume and job description go through a cleaning pipeline: lowercasing, URL/email/phone removal, stripping special characters, and collapsing whitespace. A second pass removes English stopwords (common words like "the", "and", "is") using NLTK's curated list. This ensures the analysis focuses on meaningful content words.

### Stage 3 — Skill Extraction (Two-Pass)

Skills are identified using two complementary methods:

- **Dictionary lookup:** A curated set of ~200 tech/business skills is matched against the cleaned text using word-boundary regex. This catches known terms like "python", "machine learning", "kubernetes", etc.
- **spaCy NER (Named Entity Recognition):** The `en_core_web_sm` model scans the original text for entities labeled as ORG, PRODUCT, or GPE — these often correspond to tools, frameworks, and platforms that aren't in the dictionary.

The union of both passes gives a robust skill set for each document.

### Stage 4 — TF-IDF Matching

Both documents are converted into TF-IDF (Term Frequency–Inverse Document Frequency) vectors using scikit-learn. TF-IDF assigns higher weight to words that are distinctive to a document rather than common across both. The cosine similarity between the two vectors produces a score from 0 to 1, which is displayed as a percentage. This measures overall textual alignment, not just keyword overlap.

### Stage 5 — Gap Analysis & Suggestions

The app computes set differences between JD skills and resume skills to find gaps. A rule-based engine then inspects the resume for common weaknesses: missing quantified achievements (numbers, percentages), weak action verbs, absence of a Projects or Summary section, and low keyword overlap. Each detected issue generates a specific, actionable suggestion.

---

## Project Structure

```
cognicv/
├── app.py              # Complete application (single file)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Tech Stack

| Component          | Tool           |
|--------------------|----------------|
| Web UI             | Streamlit      |
| NLP / NER          | spaCy          |
| Vectorization      | scikit-learn   |
| PDF Parsing        | PyPDF2         |
| Stopwords          | NLTK           |

---

## License

MIT — use freely for learning, portfolios, and projects.
