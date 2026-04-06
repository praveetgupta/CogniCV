"""
╔══════════════════════════════════════════════════════════════╗
║  CogniCV — Smart Resume Analyzer with Job Matching          ║
║  A Streamlit web app for resume ↔ job description analysis  ║
╚══════════════════════════════════════════════════════════════╝

Required pip installs:
    pip install streamlit spacy scikit-learn pypdf2 nltk
    python -m spacy download en_core_web_sm
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import re
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

try:
    from PyPDF2 import PdfReader
except ImportError:
    from pypdf import PdfReader


# ─────────────────────────────────────────────
# 2. ONE-TIME SETUP (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    """Load the FULL spaCy model — not blank — so NER + lemmatizer work."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.warning(
            "⚠️ spaCy model `en_core_web_sm` not found. "
            "Run `python -m spacy download en_core_web_sm` for best results. "
            "Falling back to basic tokenizer."
        )
        return spacy.blank("en")


@st.cache_resource
def download_stopwords():
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))


NLP = load_nlp()
STOP_WORDS = download_stopwords()


# ─────────────────────────────────────────────
# 3. SYNONYM MAP
# ─────────────────────────────────────────────
# Maps long-form phrases → canonical short form.
# Used ONLY during comparison — we normalize both sides equally.
SYNONYMS = {
    "natural language processing": "nlp",
    "retrieval augmented generation": "rag",
    "retrieval-augmented generation": "rag",
    "machine learning": "ml",
    "deep learning": "dl",
    "large language model": "llm",
    "large language models": "llm",
    "restful api": "rest api",
    "hugging face": "huggingface",
    # Common variations
    "scikit-learn": "sklearn",
    "scikit learn": "sklearn",
    "sci-kit learn": "sklearn",
    "postgresql": "postgres",
    "k8s": "kubernetes",
    "golang": "go",
    "node.js": "nodejs",
    "next.js": "nextjs",
    "nuxt.js": "nuxtjs",
    "ci/cd": "cicd",
    "google cloud": "gcp",
}


def normalize_skill(skill: str) -> str:
    """Reduce a skill string to its canonical form for comparison."""
    s = skill.lower().strip()
    return SYNONYMS.get(s, s)


# ─────────────────────────────────────────────
# 4. CURATED SKILL DICTIONARY
# ─────────────────────────────────────────────
KNOWN_SKILLS = {
    # Programming & scripting
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",
    "bash", "shell", "powershell", "sql", "nosql", "graphql", "html", "css",
    "sass", "less",
    # Frameworks & libraries
    "react", "angular", "vue", "svelte", "next.js", "nuxt.js", "django",
    "flask", "fastapi", "spring", "express", "node.js", "nodejs", ".net",
    "dotnet", "rails", "laravel", "streamlit", "gradio", "jquery",
    "bootstrap", "tailwind", "material ui",
    # Data / ML / AI
    "machine learning", "ml", "deep learning", "dl",
    "natural language processing", "nlp",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "hugging face", "huggingface", "transformers", "langchain", "openai",
    "llm", "large language model", "generative ai", "rag",
    "retrieval augmented generation", "fine-tuning", "bert", "gpt",
    "data science", "data analysis", "data engineering", "data visualization",
    "statistics", "a/b testing", "hypothesis testing", "regression",
    "classification", "clustering", "time series", "feature engineering",
    "mlops",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "ci/cd", "cicd",
    "linux", "unix", "nginx", "apache",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "sql server", "firebase",
    "supabase", "neo4j",
    # Tools & practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "figma", "sketch", "adobe xd", "photoshop", "illustrator",
    "agile", "scrum", "kanban", "devops", "microservices", "rest",
    "rest api", "api", "grpc", "oauth", "jwt",
    # Soft skills
    "leadership", "communication", "teamwork", "problem solving",
    "project management", "product management", "stakeholder management",
    "mentoring", "strategic thinking", "analytical thinking",
    "critical thinking", "time management", "cross-functional",
    # Certifications
    "pmp", "aws certified", "azure certified", "google certified",
    "scrum master", "six sigma", "itil",
    # Other domains
    "blockchain", "web3", "solidity", "cybersecurity", "penetration testing",
    "networking", "tcp/ip", "dns", "seo", "sem", "google analytics",
    "tableau", "power bi", "excel", "looker", "dbt", "airflow", "spark",
    "hadoop", "kafka", "etl", "data warehouse", "snowflake", "redshift",
    "bigquery",
}

# Skills too generic to flag as "missing" — they just add noise
SOFT_GENERIC_SKILLS = {
    "communication", "teamwork", "problem solving", "analytical thinking",
    "leadership", "cross-functional", "strategic thinking", "critical thinking",
    "time management", "mentoring", "stakeholder management",
}


# ─────────────────────────────────────────────
# 5. TEXT EXTRACTION & PREPROCESSING
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """Read all pages from an uploaded PDF and return combined text."""
    reader = PdfReader(uploaded_file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def preprocess_for_tfidf(text: str) -> str:
    """
    Light preprocessing for TF-IDF: lowercase, strip noise, remove stopwords.
    Does NOT run spaCy — that would be expensive and unnecessary here.
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", " ", text)
    text = re.sub(r"[^\w\s\-\.\+#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


# ─────────────────────────────────────────────
# 6. SKILL EXTRACTION
# ─────────────────────────────────────────────
def extract_skills(text: str) -> set:
    """
    Two-pass skill extraction:
      Pass 1 — Dictionary: regex-match every known skill phrase.
      Pass 2 — spaCy NER: pull ORG / PRODUCT entities as candidate skills.
    Returns raw skill strings (not yet normalized).
    """
    clean = text.lower()
    found = set()

    # Pass 1 — dictionary lookup
    for skill in KNOWN_SKILLS:
        if len(skill) <= 2:
            pattern = rf"\b{re.escape(skill)}\b"
        else:
            pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, clean):
            found.add(skill)

    # Pass 2 — spaCy NER (only if model has a pipeline)
    # IMPORTANT: Only accept NER entities that look like real tech skills.
    # Raw NER produces too many false positives (school names, event names,
    # locations, abbreviations) so we validate against KNOWN_SKILLS or
    # require the entity to be a single recognizable tech-like token.
    if NLP.pipe_names:
        doc = NLP(text[:100_000])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT"):
                ent_text = ent.text.lower().strip()
                # Only accept if it's already in our dictionary (catches
                # casing variants like "PyTorch" → "pytorch") — don't
                # blindly trust spaCy for unknown entities.
                if ent_text in KNOWN_SKILLS:
                    found.add(ent_text)

    return found


def normalize_skill_set(skills: set) -> dict:
    """
    Normalize a set of raw skills into {canonical_form: display_form}.
    Merges synonyms so "nlp" and "natural language processing" become one entry.
    Keeps the longer / more readable form as display name.
    """
    normalized = {}
    for skill in skills:
        canon = normalize_skill(skill)
        if canon not in normalized or len(skill) > len(normalized[canon]):
            normalized[canon] = skill
    return normalized


# ─────────────────────────────────────────────
# 7. MATCHING
# ─────────────────────────────────────────────
def compute_match_score(
    resume_text: str,
    jd_text: str,
    resume_canon: set,
    jd_canon: set,
) -> tuple:
    """
    Blended score from two signals:
      • Skill overlap  — what % of JD skills appear in the resume
      • TF-IDF cosine  — full-text semantic alignment

    The blend adapts to JD length:
      Short JDs (< 100 words) → mostly skill overlap (TF-IDF is noisy)
      Long JDs  (300+ words)  → balanced 55/45

    TF-IDF raw cosine between a multi-page resume and a short JD is
    structurally low (typically 5–20%) because the documents differ so
    much in length and vocabulary density.  We rescale it so that a
    cosine of ~0.30 maps to 100% (empirical ceiling for this task).

    Returns (final_score, skill_pct, text_pct) — all 0–100 floats.
    """
    # ── Skill score ──
    matched = resume_canon & jd_canon
    skill_pct = (len(matched) / len(jd_canon) * 100) if jd_canon else 0.0

    # ── Text score (rescaled) ──
    resume_clean = preprocess_for_tfidf(resume_text)
    jd_clean = preprocess_for_tfidf(jd_text)

    if not resume_clean.strip() or not jd_clean.strip():
        text_pct = 0.0
    else:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            tfidf = vectorizer.fit_transform([resume_clean, jd_clean])
            raw_cosine = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            # Rescale: 0.30 raw cosine → 100%.  Clamp to [0, 100].
            text_pct = min(raw_cosine / 0.30 * 100, 100.0)
        except Exception:
            text_pct = 0.0

    # ── Adaptive blend based on JD length ──
    jd_word_count = len(jd_text.split())
    if jd_word_count < 100:
        skill_w, text_w = 0.80, 0.20     # short JD — TF-IDF unreliable
    elif jd_word_count < 300:
        skill_w, text_w = 0.65, 0.35
    else:
        skill_w, text_w = 0.55, 0.45     # long JD — TF-IDF is meaningful

    final = skill_w * skill_pct + text_w * text_pct
    return round(final, 1), round(skill_pct, 1), round(text_pct, 1)


# ─────────────────────────────────────────────
# 8. IMPROVEMENT SUGGESTIONS (rule-based)
# ─────────────────────────────────────────────
def generate_suggestions(
    resume_text: str,
    missing_display: list,
    match_score: float,
) -> list:
    """
    Actionable suggestions via simple string checks on raw resume text.
    No heavy NLP — just pattern matching.
    """
    tips = []
    lower = resume_text.lower()

    # Missing skills
    if missing_display:
        top = missing_display[:8]
        tips.append(
            "**Add missing skills:** Consider gaining experience in or "
            f"highlighting these — {', '.join(top)}."
        )

    # Quantified achievements
    if not re.search(r"\d+\s*%|\$\s*[\d,]+|\b\d{2,}\b", lower):
        tips.append(
            "**Quantify achievements:** Use concrete numbers. "
            '*"Reduced latency by 35 %"* beats *"Improved performance."*'
        )

    # Action verbs
    strong_verbs = {
        "led", "designed", "developed", "implemented", "architected",
        "optimized", "automated", "delivered", "managed", "launched",
        "built", "created", "spearheaded", "orchestrated", "streamlined",
    }
    if len({v for v in strong_verbs if v in lower}) < 3:
        tips.append(
            "**Use stronger action verbs:** Start bullets with words "
            "like *Led, Architected, Optimized, Automated, Delivered*."
        )

    # Projects section
    if "project" not in lower:
        tips.append(
            "**Add a Projects section:** 2–3 relevant projects "
            "demonstrating the role's key skills make a strong impression."
        )

    # Professional summary
    if not any(kw in lower for kw in ("summary", "objective", "profile", "about me")):
        tips.append(
            "**Add a Professional Summary:** A 2–3 sentence overview "
            "at the top helps recruiters quickly gauge fit."
        )

    # Certifications
    if not any(kw in lower for kw in ("certified", "certification", "certificate")):
        tips.append(
            "**Consider certifications:** AWS, GCP, PMP, or domain-specific "
            "certs strengthen your profile for competitive roles."
        )

    # Low keyword overlap
    if match_score < 40:
        tips.append(
            "**Mirror the JD language:** Your resume shares very few keywords "
            "with the job post. Naturally weave in its terminology."
        )

    # Length checks
    word_count = len(lower.split())
    if word_count < 150:
        tips.append(
            "**Expand your resume:** It appears quite short. Aim for at least "
            "one full page with detailed experience bullets."
        )
    elif word_count > 1100:
        tips.append(
            "**Trim for conciseness:** For most roles 1–2 pages is ideal — "
            "prioritize recent, relevant experience."
        )

    return tips


# ─────────────────────────────────────────────
# 9. SKILL GROUPING FOR DISPLAY
# ─────────────────────────────────────────────
SKILL_CATEGORIES = {
    "AI / ML": {
        "ml", "dl", "nlp", "rag", "llm", "machine learning", "deep learning",
        "natural language processing", "computer vision", "tensorflow",
        "pytorch", "keras", "scikit-learn", "transformers", "langchain",
        "openai", "generative ai", "bert", "gpt", "fine-tuning",
        "huggingface", "hugging face", "classification", "regression",
        "clustering", "feature engineering", "mlops",
    },
    "Data": {
        "data science", "data analysis", "data engineering",
        "data visualization", "statistics", "a/b testing",
        "hypothesis testing", "time series", "pandas", "numpy", "scipy",
        "matplotlib", "seaborn", "plotly", "tableau", "power bi", "excel",
        "looker", "dbt", "airflow", "spark", "hadoop", "kafka", "etl",
        "data warehouse", "snowflake", "redshift", "bigquery",
    },
    "Cloud / DevOps": {
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "k8s", "terraform", "ansible", "jenkins", "github actions",
        "ci/cd", "cicd", "devops", "linux", "unix", "nginx", "apache",
    },
    "Backend / API": {
        "api", "rest", "rest api", "grpc", "oauth", "jwt",
        "microservices", "fastapi", "django", "flask", "spring",
        "express", "node.js", "nodejs", "rails", "laravel", ".net", "dotnet",
    },
    "Frontend": {
        "react", "angular", "vue", "svelte", "next.js", "nuxt.js",
        "html", "css", "sass", "less", "jquery", "bootstrap", "tailwind",
        "material ui", "figma", "sketch", "adobe xd",
    },
    "Databases": {
        "sql", "nosql", "mysql", "postgresql", "postgres", "mongodb",
        "redis", "elasticsearch", "cassandra", "dynamodb", "sqlite",
        "oracle", "sql server", "firebase", "supabase", "neo4j", "graphql",
    },
    "Security": {
        "cybersecurity", "penetration testing", "networking", "tcp/ip", "dns",
    },
    "Languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
        "matlab", "perl", "bash", "shell", "powershell",
    },
}


def group_skills(skills: list) -> dict:
    """Group a list of skill display names into categories."""
    grouped = {}
    uncategorized = []
    for skill in skills:
        canon = normalize_skill(skill)
        placed = False
        for category, members in SKILL_CATEGORIES.items():
            if canon in members or skill.lower() in members:
                grouped.setdefault(category, []).append(skill)
                placed = True
                break
        if not placed:
            uncategorized.append(skill)
    if uncategorized:
        grouped["Other"] = uncategorized
    return grouped


# ─────────────────────────────────────────────
# 10. STREAMLIT UI
# ─────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 70:
        return "#22c55e"
    if score >= 40:
        return "#eab308"
    return "#ef4444"


def score_label(score: float) -> str:
    if score >= 75:
        return "Excellent Fit"
    if score >= 55:
        return "Good Potential"
    if score >= 35:
        return "Partial Match"
    return "Needs Improvement"


def main():
    st.set_page_config(
        page_title="CogniCV — Smart Resume Analyzer",
        page_icon="🧠",
        layout="wide",
    )

    # ── CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="stApp"] { font-family: 'DM Sans', sans-serif; }
    .block-container { max-width: 1100px; padding-top: 2rem; }

    .cogni-header { text-align: center; padding: 2rem 0 1.2rem; }
    .cogni-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: .25rem;
    }
    .cogni-header p { font-size: 1.05rem; opacity: .65; margin-top: 0; }

    .score-card { text-align: center; padding: 1.5rem 0; }
    .score-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem; font-weight: 700; line-height: 1;
    }
    .score-label { font-size: 1.1rem; font-weight: 500; margin-top: .3rem; }
    .score-sub {
        text-align: center; margin-top: 8px;
        font-size: 0.92rem; opacity: 0.75;
    }

    .skill-pill {
        display: inline-block; padding: 5px 14px; border-radius: 999px;
        font-size: .85rem; font-weight: 500; margin: 4px;
    }
    .pill-match  { background: rgba(34,197,94,.15); color: #16a34a; border: 1px solid rgba(34,197,94,.3); }
    .pill-missing { background: rgba(239,68,68,.12); color: #dc2626; border: 1px solid rgba(239,68,68,.25); }
    .pill-resume  { background: rgba(99,102,241,.12); color: #6366f1; border: 1px solid rgba(99,102,241,.25); }

    .suggestion-box {
        background: rgba(99,102,241,.06);
        border-left: 4px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 18px; margin-bottom: 10px; font-size: .95rem;
    }
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem; font-weight: 600;
        margin: 1.6rem 0 .6rem; padding-bottom: .4rem;
        border-bottom: 2px solid rgba(99,102,241,.25);
    }
    .category-label {
        font-size: 0.82rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.05em; opacity: 0.55; margin: 10px 0 2px 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown(
        '<div class="cogni-header">'
        "<h1>🧠 CogniCV</h1>"
        "<p>Smart Resume Analyzer &amp; Job Matching</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Inputs ──
    col_resume, col_jd = st.columns(2, gap="large")
    with col_resume:
        st.markdown("##### 📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "PDF or TXT", type=["pdf", "txt"], label_visibility="collapsed",
        )
    with col_jd:
        st.markdown("##### 💼 Paste Job Description")
        job_description = st.text_area(
            "Paste here", height=220, label_visibility="collapsed",
            placeholder="Paste the full job description here …",
        )

    st.markdown("")
    analyze = st.button("🔍  Analyze Match", use_container_width=True, type="primary")

    # ─────────────────────────────────────────
    # ANALYSIS PIPELINE
    # ─────────────────────────────────────────
    if not analyze:
        return

    if not uploaded_file:
        st.warning("Please upload a resume file.")
        return
    if not job_description or len(job_description.strip()) < 30:
        st.warning("Please paste a meaningful job description (at least a few sentences).")
        return

    with st.spinner("Analyzing your resume …"):
        # A — Extract resume text
        if uploaded_file.name.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

        if len(resume_text.strip()) < 20:
            st.error("Could not extract enough text. Try a different file or paste as TXT.")
            return

        # B — Extract raw skills from both documents
        resume_skills_raw = extract_skills(resume_text)
        jd_skills_raw = extract_skills(job_description)

        # C — Normalize for comparison (merge synonyms)
        resume_norm = normalize_skill_set(resume_skills_raw)
        jd_norm = normalize_skill_set(jd_skills_raw)

        resume_canon = set(resume_norm.keys())
        jd_canon = set(jd_norm.keys())

        # D — Compute scores
        match_score, skill_pct, text_pct = compute_match_score(
            resume_text, job_description, resume_canon, jd_canon,
        )

        # E — Matched / missing / extra
        matched_canon = resume_canon & jd_canon
        missing_canon = jd_canon - resume_canon
        extra_canon = resume_canon - jd_canon

        # Filter generic/soft skills from "missing" — they're noise
        missing_canon = {
            s for s in missing_canon
            if s not in SOFT_GENERIC_SKILLS and len(s) > 2
        }

        # Map canonical keys back to human-readable display names
        matched_display = sorted(jd_norm[c] for c in matched_canon)
        missing_display = sorted(jd_norm[c] for c in missing_canon)
        extra_display = sorted(
            resume_norm[c] for c in extra_canon
            if c not in SOFT_GENERIC_SKILLS
        )

        # F — Generate suggestions
        suggestions = generate_suggestions(resume_text, missing_display, match_score)

    # ─────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────
    st.markdown("---")

    # Score display
    color = score_color(match_score)
    label = score_label(match_score)

    st.markdown(
        f'<div class="score-card">'
        f'<div class="score-number" style="color:{color}">{match_score}%</div>'
        f'<div class="score-label" style="color:{color}">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(min(match_score / 100, 1.0))

    color_skill = "#22c55e" if skill_pct >= 60 else ("#eab308" if skill_pct >= 35 else "#ef4444")
    color_text = "#22c55e" if text_pct >= 40 else ("#eab308" if text_pct >= 20 else "#ef4444")
    st.markdown(
        f'<div class="score-sub">'
        f'Skill Overlap: <b style="color:{color_skill}">{skill_pct}%</b>'
        f' &nbsp;·&nbsp; '
        f'Text Similarity: <b style="color:{color_text}">{text_pct}%</b>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Matched keywords
    st.markdown('<div class="section-title">✅ Matched Keywords</div>', unsafe_allow_html=True)
    if matched_display:
        pills = "".join(
            f'<span class="skill-pill pill-match">{s}</span>' for s in matched_display
        )
        st.markdown(pills, unsafe_allow_html=True)
    else:
        st.info("No overlapping skill keywords detected.")

    # Missing skills (grouped by category)
    st.markdown('<div class="section-title">❌ Missing Skills</div>', unsafe_allow_html=True)
    if missing_display:
        grouped = group_skills(missing_display)
        for category, skills in grouped.items():
            st.markdown(
                f'<div class="category-label">{category}</div>',
                unsafe_allow_html=True,
            )
            pills = "".join(
                f'<span class="skill-pill pill-missing">{s}</span>' for s in sorted(skills)
            )
            st.markdown(pills, unsafe_allow_html=True)
    else:
        st.success("Your resume covers all detected JD skills!")

    # Extra resume skills
    if extra_display:
        st.markdown(
            '<div class="section-title">🔵 Additional Resume Skills</div>',
            unsafe_allow_html=True,
        )
        pills = "".join(
            f'<span class="skill-pill pill-resume">{s}</span>' for s in extra_display
        )
        st.markdown(pills, unsafe_allow_html=True)

    # Improvement suggestions
    st.markdown(
        '<div class="section-title">💡 Improvement Suggestions</div>',
        unsafe_allow_html=True,
    )
    if suggestions:
        for tip in suggestions:
            st.markdown(
                f'<div class="suggestion-box">{tip}</div>', unsafe_allow_html=True,
            )
    else:
        st.success("Your resume looks solid! Keep iterating as you grow.")

    # Debug expander
    with st.expander("🔬 Debug — All Extracted Skills"):
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("**Resume (raw → canonical)**")
            for canon, display in sorted(resume_norm.items()):
                st.text(f"  {display} → {canon}")
        with dcol2:
            st.markdown("**JD (raw → canonical)**")
            for canon, display in sorted(jd_norm.items()):
                st.text(f"  {display} → {canon}")


if __name__ == "__main__":
    main()