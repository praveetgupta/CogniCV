# 🧠 CogniCV — Batch Resume Screening & Candidate Ranking

CogniCV helps recruiters screen a stack of resumes against a job description in seconds. Upload a batch of CVs (PDF / DOCX / TXT), paste the JD, and get a **ranked shortlist** with per-candidate skill-gap analysis, experience and education checks, screening flags, and CSV export — all running locally, with no resume data leaving your machine.

```
JD + resumes  ─▶  parse  ─▶  extract skills / experience / education
                              │
                              ▼
              must-have vs nice-to-have requirement analysis
                              │
                              ▼
        blended score:  skills · semantic fit · experience · education
                              │
                              ▼
            ranked shortlist + gap analysis + CSV export
```

---

## Quick start

```bash
# 1. Install (Python 3.9+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Launch the web app
streamlit run app.py
```

Then in the browser: paste a job description, upload resumes, done. Try it immediately with the files in [`sample_data/`](sample_data/).

### CLI (for pipelines / bulk runs)

```bash
python -m cognicv --jd sample_data/job_description.txt \
                  --resumes sample_data/ \
                  --csv results.csv --top 10
```

---

## Features

| | |
|---|---|
| **Batch ranking** | Score dozens of resumes against one JD at once; results in a sortable table with CSV export. |
| **Requirement analysis** | The JD is parsed into **must-have** (weighted 3×) and **nice-to-have** skills using its own section structure ("Requirements:" vs "Nice to have:") and inline cues ("X is a plus"). You can adjust the detected must-haves before scoring. |
| **Skill intelligence** | ~350-skill taxonomy across 15 categories with synonym normalization (K8s = Kubernetes, golang = Go) and disambiguation for tricky tokens — "Go" the language vs "go the extra mile", "Spring" the framework vs "Spring 2024", "Excel" the tool vs "excels at". |
| **Experience & education** | Estimates years of experience by merging employment date ranges (overlapping jobs aren't double-counted) and compares against the JD's "N+ years" requirement; detects highest degree earned. |
| **Semantic fit** | Full-text similarity beyond keywords. TF-IDF out of the box; installs `sentence-transformers` → automatic upgrade to embedding-based matching. |
| **AI deep review** (optional) | Claude reads each top-ranked resume against the JD and judges the *evidence* behind claimed skills — catching what keyword matching can't ("no ML experience yet" still contains "ML"). Returns a fit score, evidence-backed vs unsupported skills, strengths, and concerns per candidate. |
| **Screening flags** | Surfaces "missing 3 must-haves", "experience couldn't be estimated", "possibly a scanned PDF" — so unknowns get human review instead of silent penalties. |
| **Blind screening** | One-click anonymization hides names and contact details during the first pass to reduce unconscious bias. |
| **Hard filtering** | Optionally exclude candidates missing more than N must-have skills. |

---

## How scoring works

Each candidate gets a 0–100 score blending four components (weights adjustable in the sidebar):

| Component | Default weight | What it measures |
|---|---|---|
| Skill coverage | 45% | Weighted % of JD skills present — must-haves count 3× nice-to-haves |
| Semantic fit | 25% | Full-text alignment between resume and JD (TF-IDF or embeddings) |
| Experience | 20% | Estimated years vs the JD's requirement (linear up to 100%) |
| Education | 10% | Degree level vs the JD's requirement |

Two fairness rules are built in:

1. **No phantom requirements.** If the JD doesn't specify years of experience or a degree, that component is dropped and the remaining weights renormalized.
2. **Unknown ≠ unqualified.** If a resume's dates can't be parsed, the candidate isn't zeroed out — the component is excluded and a ⚠️ flag tells the reviewer to check manually.

Generic soft skills ("communication", "teamwork") are detected but excluded from requirements and gap lists — every resume claims them, so they carry no signal.

---

## Project structure

```
CogniCV/
├── app.py                  # Streamlit web UI
├── cognicv/                # Engine (importable, UI-independent)
│   ├── taxonomy.py         #   skill dictionary: canonicals, aliases, categories
│   ├── skills.py           #   regex skill extraction w/ disambiguation
│   ├── extraction.py       #   PDF / DOCX / TXT → text
│   ├── candidate.py        #   name, contact, experience, education parsing
│   ├── jd.py               #   must-have / nice-to-have requirement parsing
│   ├── semantic.py         #   TF-IDF + optional embeddings similarity
│   ├── scoring.py          #   blended scoring & ranking
│   └── cli.py              #   command-line batch screening
├── tests/                  # 80 tests (pytest)
├── sample_data/            # demo JD + 4 fake resumes
└── requirements.txt
```

The engine has no Streamlit dependency — use it directly:

```python
from cognicv import extract_text, parse_candidate, parse_jd, score_candidates

jd = parse_jd(open("jd.txt").read())
profile = parse_candidate("cv.pdf", extract_text("cv.pdf", open("cv.pdf", "rb").read()))
[result] = score_candidates([profile], jd)
print(result.total, result.missing_must, result.flags)
```

---

## Optional upgrades

```bash
pip install anthropic               # AI deep review with Claude
pip install sentence-transformers   # meaning-aware semantic matching (~90MB model)
pip install pdfplumber              # fallback extractor for stubborn PDF layouts
```

All are detected automatically at runtime — no configuration needed.

### AI deep review (Claude)

The deterministic score ranks candidates; the AI review is a second opinion on
the top of the list. Claude (`claude-opus-4-8`) reads the full resume against
the JD and reports which required skills have real supporting evidence
(projects, roles, metrics) versus bare mentions, plus concrete strengths and
concerns. The deterministic ranking is never silently altered — AI results
appear as an extra column and per-candidate panel.

Setup: `pip install anthropic`, then provide credentials via the
`ANTHROPIC_API_KEY` environment variable, `ant auth login`, or the API-key
field in the app. In the web app, use the **🤖 AI deep review** panel; from the
CLI:

```bash
python -m cognicv --jd jd.txt --resumes ./resumes/ --llm-top 5
```

Notes: resume text is sent to the Anthropic API only when this feature is
invoked; typical cost is a few cents per candidate; the prompt instructs the
model to judge job fit only and ignore protected characteristics.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest tests/
```

---

## Limitations & responsible use

CogniCV is a **decision-support tool, not a decision-maker**:

- Skill extraction is keyword-based: it can miss skills phrased unusually and can credit skills mentioned in a negative context ("no ML experience yet" still mentions ML). The per-candidate detail view exists so reviewers verify matches.
- Experience estimates come from date ranges anywhere in the resume, including education, and are labeled as estimates.
- Scanned/image-only PDFs yield little text; such candidates are flagged, not silently discarded.
- Automated resume screening may be regulated in your jurisdiction (e.g. NYC Local Law 144, EU AI Act). Keep a human in the loop for every reject decision, and audit outcomes for disparate impact across groups.

## License

MIT — see [LICENSE](LICENSE).
