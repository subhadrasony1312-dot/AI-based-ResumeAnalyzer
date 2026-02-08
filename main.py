import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# =====================
# DATABASE CONFIG
# =====================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "sb@19092006",
    "database": "resume_analyzer"
}

# =====================
# NLTK SETUP
# =====================
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# =====================
# DATABASE FUNCTIONS
# =====================
import os

def init_db():
    # If running on Streamlit Cloud, skip DB
    if os.getenv("STREAMLIT_RUNTIME"):
        return None

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                resume_name VARCHAR(255),
                match_score DECIMAL(5,2),
                job_description TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_resume TEXT
            )
        """)
        conn.commit()
        cursor.close()
        return conn
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return None


def save_to_db(names, scores, job_desc, processed_texts):
    conn = init_db()
    if not conn:
        return
    cursor = conn.cursor()
    for n, s, t in zip(names, scores, processed_texts):
        cursor.execute("""
            INSERT INTO analysis_results
            (resume_name, match_score, job_description, processed_resume)
            VALUES (%s, %s, %s, %s)
        """, (n, s, job_desc[:1000], t[:1000]))
    conn.commit()
    cursor.close()
    conn.close()

def get_db_stats():
    conn = init_db()
    if not conn:
        return pd.DataFrame()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT resume_name, match_score, analysis_date
        FROM analysis_results
        ORDER BY analysis_date DESC
        LIMIT 10
    """)
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(data, columns=["Resume", "Score", "Date"])


st.markdown("""
            Upload resume (pdf) and paste the job description to see how well they match !
            This tool uses **TF-IDF + Cosine Similarity** to analyze resumes against job requirements.
            """)
with st.sidebar:
    st.header("About")
    st.info("""
    *Features:*
    - Measures how your resume matches a job description
    - Upload 10+ resumes at once
    - Color-coded match score(🔴Low 🟡Medium 🟢High )
    - Side-by-side comparison table
    - Individual suggetions
    - Store the data in database
            """)
    st.header("How It works")
    st.write("""
    1. Upload your resume (PDF)
    2. Paste the job description
    3. Click **Analyze Match**
    4. Review score & suggetion
    """)
# =====================
# TEXT PROCESSING
# =====================
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        return " ".join([p.extract_text() or "" for p in reader.pages])
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    return " ".join([w for w in words if w not in stop_words and len(w) > 2])

def preprocess(text):
    return remove_stopwords(clean_text(text))

# =====================
# ANALYSIS FUNCTION
# =====================
def analyze_resumes(resumes, jd):
    jd_clean = preprocess(jd)
    resumes_clean = [preprocess(r) for r in resumes]

    corpus = resumes_clean + [jd_clean]
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf = vectorizer.fit_transform(corpus)

    scores = cosine_similarity(tfidf[:-1], tfidf[-1]).flatten() * 100

    jd_terms = set(jd_clean.split())
    gaps = []
    for r in resumes_clean:
        r_terms = set(r.split())
        gaps.append(list(jd_terms - r_terms)[:10])

    return [round(float(s), 2) for s in scores], gaps, resumes_clean

# =====================
# STREAMLIT APP
# =====================
def main():
    st.set_page_config("AI Resume Analyzer", "📊", layout="wide")
    st.title("🤖 AI-Based Resume Analyzer")

    tab1, tab2 = st.tabs(["📊 Analyze", "⌛ History"])

    # =====================
    # TAB 1 — ANALYZE
    # =====================
    with tab1:
        uploaded_files = st.file_uploader(
            "📁 Upload multiple resumes (PDF)",
            type=["pdf"],
            accept_multiple_files=True
        )

        job_description = st.text_area(
            "📝 Paste job description",
            height=200
        )

        if st.button("🚀 Analyze All", type="primary"):
            if not uploaded_files or not job_description.strip():
                st.warning("Please upload resumes and paste job description")
                return

            texts, names = [], []
            for f in uploaded_files:
                t = extract_text_from_pdf(f)
                if t:
                    texts.append(t)
                    names.append(f.name)

            scores, gaps, processed = analyze_resumes(texts, job_description)

            df = pd.DataFrame({
                "Resume": names,
                "Match Score (%)": scores,
                "Missing Skills": [", ".join(g) if g else "None" for g in gaps]
            }).sort_values("Match Score (%)", ascending=False)

            st.session_state["df"] = df
            save_to_db(names, scores, job_description, processed)

            st.subheader("📊 Results")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Results (CSV)",
                df.to_csv(index=False),
                "resume_analysis.csv",
                "text/csv"
            )

            # Graph
            fig, ax = plt.subplots(figsize=(10, len(df)*0.6 + 1))
            colors = [
                "#ff4b4b" if s < 40 else "#ffa726" if s < 70 else "#0f9d58"
                for s in df["Match Score (%)"]
            ]
            bars = ax.barh(df["Resume"], df["Match Score (%)"], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Match Score (%)")
            ax.set_title("Resume vs Job Match")

            for bar, score in zip(bars, df["Match Score (%)"]):
                ax.text(
                    score + 1,
                    bar.get_y() + bar.get_height()/2,
                    f"{score}%",
                    va="center",
                    fontweight="bold"
                )

            st.pyplot(fig)

            # Suggestions
            st.subheader("💡 Suggestions")
            for _, r in df.iterrows():
                if r["Match Score (%)"] < 40:
                    st.error(f"🔴 {r['Resume']} – Needs major improvement")
                elif r["Match Score (%)"] < 70:
                    st.warning(f"🟡 {r['Resume']} – Moderate match")
                else:
                    st.success(f"🟢 {r['Resume']} – Excellent match")

    # =====================
    # TAB 2 — HISTORY ONLY
    # =====================
    with tab2:
        st.subheader("⌛ Previous Analysis")
        st.dataframe(get_db_stats(), use_container_width=True)

# =====================
# RUN APP
# =====================
if __name__ == "__main__":
    main()

