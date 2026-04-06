"""
╔══════════════════════════════════════════════════════════════╗
║  CogniCV — Smart Resume Analyzer with Job Matching          ║
║  A Streamlit web app for resume ↔ job description analysis  ║
╚══════════════════════════════════════════════════════════════╝

Required pip installs:
    pip install streamlit spacy scikit-learn pypdf2 nltk
    python -m spacy download en_core_web_sm
"""

SYNONYMS = {
    "nlp": ["natural language processing"],
    "rag": ["retrieval augmented generation", "retrieval-augmented generation"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "llm": ["large language models"],
    "api": ["rest", "rest api", "restful api"],
    "huggingface": ["transformers"],
}

def apply_synonyms(text: str) -> str:
    text = text.lower()
    for key, values in SYNONYMS.items():
        for val in values:
            text = text.replace(val, key)
    return text

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import re
import string
import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# PDF text extraction
try:
    from PyPDF2 import PdfReader          # PyPDF2 >= 3.x
except ImportError:
    from pypdf import PdfReader           # fallback to pypdf

# ─────────────────────────────────────────────
# 2. ONE-TIME SETUP (cached so it only runs once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_nlp():
    """Load spaCy English model once and cache it."""
    import spacy
    return spacy.blank("en")

@st.cache_resource
def download_stopwords():
    """Download NLTK stopwords once and return the set."""
    nltk.download("stopwords", quiet=True)
    return set(stopwords.words("english"))

NLP = load_nlp()
STOP_WORDS = download_stopwords()

# ─────────────────────────────────────────────
# 3. CURATED SKILL DICTIONARIES
# ─────────────────────────────────────────────
# A broad set of recognizable tech / business skills so that
# extraction is not solely dependent on spaCy NER.
CORE_SKILLS = {
        "python", "machine learning", "nlp", "deep learning",
        "pytorch", "tensorflow", "rag"
    }    

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
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "hugging face", "transformers", "langchain", "openai", "llm",
    "large language model", "generative ai", "rag",
    "retrieval augmented generation", "fine-tuning", "bert", "gpt",
    "data science", "data analysis", "data engineering", "data visualization",
    "statistics", "a/b testing", "hypothesis testing", "regression",
    "classification", "clustering", "time series", "feature engineering",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "ci/cd", "cicd",
    "linux", "unix", "nginx", "apache",
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "sqlite", "oracle", "sql server", "firebase",
    "supabase", "neo4j",
    # Tools & practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack",
    "figma", "sketch", "adobe xd", "photoshop", "illustrator",
    "agile", "scrum", "kanban", "devops", "microservices", "rest", "restful",
    "api", "graphql", "grpc", "oauth", "jwt",
    # Soft skills & business
    "leadership", "communication", "teamwork", "problem solving",
    "project management", "product management", "stakeholder management",
    "presentation", "mentoring", "strategic thinking", "analytical thinking",
    "critical thinking", "time management", "cross-functional",
    # Certifications & standards
    "pmp", "aws certified", "azure certified", "google certified",
    "scrum master", "six sigma", "itil",
    # Other domains
    "blockchain", "web3", "solidity", "cybersecurity", "penetration testing",
    "networking", "tcp/ip", "dns", "seo", "sem", "google analytics",
    "tableau", "power bi", "excel", "looker", "dbt", "airflow", "spark",
    "hadoop", "kafka", "etl", "data warehouse", "snowflake", "redshift",
    "bigquery",
}

# ─────────────────────────────────────────────
# 4. TEXT EXTRACTION & PREPROCESSING
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    """Read all pages from an uploaded PDF and return combined text."""
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def preprocess(text: str) -> str:
    text = text.lower()

    # Apply synonym normalization FIRST
    text = apply_synonyms(text)

    # Remove URLs, emails, phone numbers
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", " ", text)

    # Remove special characters
    text = re.sub(r"[^\w\s]", " ", text)

    # Collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Use spaCy (FIXED: use NLP not nlp)
    doc = NLP(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    return " ".join(tokens)


def remove_stopwords(text: str) -> str:
    """Remove English stopwords from a text string."""
    return " ".join(w for w in text.split() if w not in STOP_WORDS)

# ─────────────────────────────────────────────
# 5. SKILL EXTRACTION
# ─────────────────────────────────────────────
def extract_skills(text: str) -> set:
    """
    Two-pass skill extraction:
      1. Dictionary lookup — match every known skill phrase in text.
      2. spaCy NER — pull out entities tagged as ORG, PRODUCT, or GPE
         that look like tool / tech names (single or two-word).
    Returns a de-duplicated set of skill strings.
    """
    clean = apply_synonyms(text.lower())
    found: set[str] = set()

    # Pass 1 — dictionary
    for skill in KNOWN_SKILLS:
        # Use word-boundary regex so "r" doesn't match every word with 'r'
        if len(skill) <= 2:
            pattern = rf"\b{re.escape(skill)}\b"
        else:
            pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"
        if re.search(pattern, clean):
            found.add(skill)

    # Pass 2 — spaCy NER on *original* (uncleaned) text for better entity recognition
    doc = NLP(text)
    for ent in doc.ents:
        label = ent.label_
        ent_text = ent.text.lower().strip()
        if label in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
            # Keep it only if it looks like a tech term (≤ 4 words, no pure numbers)
            if 1 <= len(ent_text.split()) <= 4 and not ent_text.isnumeric():
                found.add(ent_text)

    return found

def filter_relevant_skills(skills: set) -> set:
    IMPORTANT = {
        "python", "machine learning", "deep learning", "nlp",
        "rag", "transformers", "pytorch", "tensorflow",
        "docker", "aws", "gcp", "mlops", "fastapi",
        "data science", "api", "classification",
        "cybersecurity", "generative ai"
    }

    return {s for s in skills if s in IMPORTANT}

# ─────────────────────────────────────────────
# 6. TF-IDF MATCHING
# ─────────────────────────────────────────────
def compute_match_score(resume_text: str, jd_text: str, resume_skills: set, jd_skills: set):
    # --- TEXT SIMILARITY ---
    resume_clean = apply_synonyms(resume_text.lower())
    jd_clean = apply_synonyms(jd_text.lower())

    # remove soft skill noise
    NOISE_WORDS = {"communication", "teamwork", "problem", "skills"}

    jd_clean = " ".join([w for w in jd_clean.split() if w not in NOISE_WORDS])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))

    # Fallback if text becomes empty after preprocessing
    if not resume_clean.strip() or not jd_clean.strip():
        text_score = 0
    else:
        try:
            tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
            text_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            text_score = 0

    # --- SKILL MATCH (better logic) ---
    matched = resume_skills & jd_skills

    core_matches = len([s for s in matched if s in CORE_SKILLS])
    other_matches = len(matched) - core_matches

    total_possible = len(jd_skills)

    if total_possible:
        skill_score = (core_matches * 2 + other_matches) / (total_possible * 1.5)
    else:
        skill_score = 0
    
    skill_score = min(skill_score, 1.0)
    # --- FINAL SCORE ---
    if skill_score > 0.6 and text_score < 0.2:
        text_score += 0.1  # small boost for realistic alignment

    final_score = (0.65 * skill_score + 0.35 * text_score) * 100

    return round(final_score, 1), round(skill_score * 100, 1), round(text_score * 100, 1)

# ─────────────────────────────────────────────
# 7. IMPROVEMENT SUGGESTIONS (rule-based)
# ─────────────────────────────────────────────
def generate_suggestions(
    resume_text: str,
    missing_skills: set,
    match_score: float,
) -> list[str]:
    """
    Produce actionable, prioritized suggestions based on the gap analysis.
    Pure rule-based — no external API needed.
    """
    tips: list[str] = []
    clean = preprocess(resume_text)

    # ── Missing skills ──
    if missing_skills:
        top = sorted(missing_skills)[:8]
        tips.append(
            "**Add missing skills:** Consider gaining experience in or "
            f"highlighting these on your resume — {', '.join(top)}."
        )
    
    if missing_skills:
        top_missing = list(missing_skills)[:3]
        tips.append(f"**Top gaps:** {', '.join(top_missing)}")

    # ── Quantifiable achievements ──
    has_numbers = bool(re.search(r"\d+\s*%|\$\s*\d|#?\d{2,}", clean))
    if not has_numbers:
        tips.append(
            "**Quantify achievements:** Use numbers and percentages. "
            'For example, *"Reduced API latency by 35 %"* is stronger '
            'than *"Improved API performance."*'
        )

    # ── Action verbs ──
    action_verbs = {
        "led", "designed", "developed", "implemented", "architected",
        "optimized", "automated", "delivered", "managed", "launched",
        "built", "created", "spearheaded", "orchestrated", "streamlined",
    }
    found_verbs = {v for v in action_verbs if v in clean}
    if len(found_verbs) < 3:
        tips.append(
            "**Use stronger action verbs:** Start bullet points with words "
            "like *Led, Designed, Architected, Optimized, Automated, Delivered*."
        )

    # ── Project section ──
    if "project" not in clean:
        tips.append(
            "**Add a Projects section:** Showcase 2–3 relevant personal or "
            "open-source projects that demonstrate the skills the role requires."
        )

    # ── Summary / objective ──
    if "summary" not in clean and "objective" not in clean and "profile" not in clean:
        tips.append(
            "**Add a Professional Summary:** A 2–3 sentence overview at the top "
            "helps recruiters quickly see your fit for the role."
        )

    # ── Certifications ──
    cert_keywords = {"certified", "certification", "certificate", "license"}
    if not cert_keywords.intersection(clean.split()):
        tips.append(
            "**Consider certifications:** Relevant certifications (AWS, PMP, "
            "Google Cloud, etc.) strengthen your profile for competitive roles."
        )

    # ── Keyword density ──
    if match_score < 40:
        tips.append(
            "**Mirror the job description language:** Your resume and the job "
            "post share very few keywords.  Re-read the JD and naturally weave "
            "its terminology into your experience bullets."
        )

    # ── Length heuristic ──
    word_count = len(clean.split())
    if word_count < 150:
        tips.append(
            "**Expand your resume:** It appears quite short.  Aim for at least "
            "one full page with detailed experience descriptions."
        )
    elif word_count > 1100:
        tips.append(
            "**Trim for conciseness:** Your resume is very long.  For most "
            "roles, 1–2 pages is ideal — prioritize recent and relevant experience."
        )

    return tips

# ─────────────────────────────────────────────
# 8. STREAMLIT UI
# ─────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 70:
        return "#22c55e"   # green
    elif score >= 40:
        return "#eab308"   # yellow (IMPORTANT CHANGE)
    else:
        return "#ef4444"   # red


def score_label(score: float) -> str:
    if score >= 75:
        return "Excellent Fit"
    elif score >= 55:
        return "Good Potential"
    elif score >= 35:
        return "Partial Match"
    else:
        return "Needs Improvement"

def group_skills(skills: set):
    groups = {
        "Cloud": {"aws", "gcp", "azure"},
        "DevOps": {"docker", "kubernetes", "mlops"},
        "Backend/API": {"api", "fastapi", "rest"},
        "AI/ML": {"classification", "deep learning", "nlp", "rag"},
        "Security": {"cybersecurity"},
    }

    grouped = {}

    for skill in skills:
        found = False
        for group, keywords in groups.items():
            if skill in keywords:
                grouped.setdefault(group, []).append(skill)
                found = True
                break
        if not found:
            grouped.setdefault("Other", []).append(skill)

    return grouped


def main():
    # ── Page config ──
    st.set_page_config(
        page_title="CogniCV — Smart Resume Analyzer",
        page_icon="🧠",
        layout="wide",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── globals ── */
    html, body, [class*="stApp"] {
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1100px; padding-top: 2rem; }

    /* ── header ── */
    .cogni-header {
        text-align: center;
        padding: 2rem 0 1.2rem;
    }
    .cogni-header h1 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: .25rem;
    }
    .cogni-header p {
        font-size: 1.05rem;
        opacity: .65;
        margin-top: 0;
    }

    /* ── score ring ── */
    .score-card {
        text-align: center;
        padding: 1.5rem 0;
    }
    .score-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-label {
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: .3rem;
    }

    /* ── pills for skills ── */
    .skill-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: .85rem;
        font-weight: 500;
        margin: 4px 4px;
    }
    .pill-match {
        background: rgba(34,197,94,.15);
        color: #16a34a;
        border: 1px solid rgba(34,197,94,.3);
    }
    .pill-missing {
        background: rgba(239,68,68,.12);
        color: #dc2626;
        border: 1px solid rgba(239,68,68,.25);
    }
    .pill-resume {
        background: rgba(99,102,241,.12);
        color: #6366f1;
        border: 1px solid rgba(99,102,241,.25);
    }

    /* ── suggestion cards ── */
    .suggestion-box {
        background: rgba(99,102,241,.06);
        border-left: 4px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 18px;
        margin-bottom: 10px;
        font-size: .95rem;
    }

    /* ── divider ── */
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 1.6rem 0 .6rem;
        padding-bottom: .4rem;
        border-bottom: 2px solid rgba(99,102,241,.25);
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

    # ── Input columns ──
    col_resume, col_jd = st.columns(2, gap="large")

    with col_resume:
        st.markdown("##### 📄 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "PDF or TXT file",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )

    with col_jd:
        st.markdown("##### 💼 Paste Job Description")
        job_description = st.text_area(
            "Paste the full job description here",
            height=220,
            label_visibility="collapsed",
            placeholder="Paste the full job description here …",
        )

    st.markdown("")  # spacer

    # ── Analyze button ──
    analyze = st.button("🔍  Analyze Match", use_container_width=True, type="primary")

    # ─────────────────────────────────────────
    # ANALYSIS PIPELINE
    # ─────────────────────────────────────────
    if analyze:
        # Validate inputs
        if not uploaded_file:
            st.warning("Please upload a resume file.")
            return
        if not job_description or len(job_description.strip()) < 30:
            st.warning("Please paste a meaningful job description (at least a few sentences).")
            return

        with st.spinner("Analyzing your resume …"):

            # Step A — Extract resume text
            if uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = uploaded_file.read().decode("utf-8", errors="ignore")

            if len(resume_text.strip()) < 20:
                st.error("Could not extract enough text from the resume. "
                         "Try a different file or paste as TXT.")
                return

            # Step B — Extract skills from both documents
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(job_description)

            # Step C — Filter important skills only
            resume_skills_filtered = resume_skills
            jd_skills_filtered = jd_skills

            # Step C — Compute match score via TF-IDF + cosine similarity
            match_score, skill_score, text_score = compute_match_score(
                resume_text,
                job_description,
                resume_skills_filtered,
                jd_skills_filtered,
            )

            # Step D — Find matched & missing skills
            matched_skills = resume_skills_filtered & jd_skills_filtered
            # Remove generic/non-skill words
            GENERIC_WORDS = {
                "communication", "teamwork", "india", "cross-functional",
                "problem solving", "analytical thinking", "leadership", 
            }

            missing_skills = {
                skill for skill in (jd_skills_filtered - resume_skills_filtered)
                if skill not in GENERIC_WORDS and len(skill) > 2
            }
            extra_skills = resume_skills - jd_skills   # skills only in resume

            # Step E — Generate suggestions
            suggestions = generate_suggestions(resume_text, missing_skills, match_score)

        # ── RESULTS ──
        st.markdown("---")

        # ── Score ──
        color = score_color(match_score)
        label = score_label(match_score)

        if skill_score == 100 and text_score < 30:
            color = "#f97316"
            label = "Strong Skills, Weak Alignment"

        st.markdown(
            f'<div class="score-card">'
            f'<div class="score-number" style="color:{color}">{match_score}%</div>'
            f'<div class="score-label" style="color:{color}">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(match_score / 100, 1.0))

        if match_score >= 75:
            explanation = "Strong match"
        elif match_score >= 55:
            explanation = "Good match with minor gaps"
        elif match_score >= 40:
            explanation = "Partial match"
        else:
            explanation = "Needs improvement"

        st.markdown(f"""
        <div style="text-align:center; margin-top:8px; font-size:0.95rem; opacity:0.75;">
        {explanation}
        </div>
        """, unsafe_allow_html=True)


        color_skill = "#22c55e" if skill_score > 70 else "#eab308"
        color_text = "#22c55e" if text_score > 50 else "#ef4444"
        st.markdown(f"""
        <div style="text-align:center; margin-top:10px; font-size:0.95rem; opacity:0.85;">
        Skill Match: <b style="color:{color_skill}">{skill_score}%</b> &nbsp;&nbsp;|&nbsp;&nbsp;
        Text Match: <b style="color:{color_text}">{text_score}%</b>
        </div>
        """, unsafe_allow_html=True)

        # ── Matched Keywords ──
        st.markdown('<div class="section-title">✅ Matched Keywords</div>', unsafe_allow_html=True)
        if matched_skills:
            pills = "".join(
                f'<span class="skill-pill pill-match">{s}</span>'
                for s in sorted(matched_skills)
            )
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.info("No overlapping skill keywords detected.")

        # ── Missing Skills ──
        st.markdown('<div class="section-title">❌ Missing Skills</div>', unsafe_allow_html=True)
        if missing_skills:
            grouped = group_skills(missing_skills)

            for group, skills in grouped.items():
                st.markdown(f"**{group}:**")
                pills = "".join(
                    f'<span class="skill-pill pill-missing">{s}</span>'
                    for s in sorted(skills)
                )
                st.markdown(pills, unsafe_allow_html=True)
        else:
            st.success("Great — your resume covers all detected JD skills!")

        # ── Extra Resume Skills (bonus context) ──
        if extra_skills:
            st.markdown(
                '<div class="section-title">🔵 Skills Found in Resume (Not in JD)</div>',
                unsafe_allow_html=True,
            )
            pills = "".join(
                f'<span class="skill-pill pill-resume">{s}</span>'
                for s in sorted(extra_skills)
            )
            st.markdown(pills, unsafe_allow_html=True)

        # ── Improvement Suggestions ──
        st.markdown(
            '<div class="section-title">💡 Improvement Suggestions</div>',
            unsafe_allow_html=True,
        )
        if suggestions:
            for tip in suggestions:
                st.markdown(
                    f'<div class="suggestion-box">{tip}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("Your resume looks solid! Keep iterating as you grow.")

        # ── Raw skill debug (collapsed) ──
        with st.expander("🔬 Debug — Extracted Skill Details"):
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                st.markdown("**Resume skills**")
                st.write(sorted(resume_skills) if resume_skills else ["(none detected)"])
            with dcol2:
                st.markdown("**JD skills**")
                st.write(sorted(jd_skills) if jd_skills else ["(none detected)"])


# ─────────────────────────────────────────────
# 9. ENTRYPOINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
