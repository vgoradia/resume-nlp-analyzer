import streamlit as st
import spacy
import textstat
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def analyze(text: str) -> dict:
    doc = nlp(text)

    # Sentences
    text_for_sentences = " ".join(line.strip() for line in text.splitlines() if line.strip())
    doc_sentences = nlp(text_for_sentences)
    sentences = list(doc_sentences.sents)

    # "Content words" 
    content_words = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]

    # All alphabetic words 
    all_words = [t.text.lower() for t in doc if t.is_alpha]

    # Average sentence length 
    avg_sentence_length = (
        sum(len(s.text.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )

    # Readability
    flesch = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    # Named entities
    entities = [(ent.text.strip().replace("\n", " "), ent.label_) for ent in doc.ents][:10]

    # Action verbs 
    bad_verbs = {"resume"} 
    action_verbs = [
        t.lemma_.lower() for t in doc
        if t.pos_ == "VERB"
        and t.is_alpha
        and not t.is_stop
        and t.lemma_.lower() not in bad_verbs
    ]

    # Weak verbs list 
    weak_verbs_set = {
        "help", "work", "learn", "do", "make", "use", "try", "assist", "handle", "support"
    }

    weak_verb_replacements = {
    "help": ["support", "enable", "facilitate"],
    "work": ["execute", "drive", "deliver"],
    "learn": ["mastered", "acquired", "developed"],
    "do": ["execute", "implement", "complete"],
    "make": ["build", "create", "develop"],
    "use": ["leverage", "utilize", "implement"],
    "try": ["pursue", "implement", "execute"],
    "assist": ["led", "drove", "spearheaded"],
    "handle": ["managed", "oversaw", "directed"],
    "support": ["enabled", "accelerated", "strengthened"]
    }

    weak_verb_hits = [(v, weak_verb_replacements.get(v, [])) for v in set(action_verbs) if v in weak_verbs_set]

    # Bullet detection
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if re.match(r"^(\-|\*|•)\s+", ln)]
    bullet_count = len(bullet_lines)

    # Resume signal metrics
    total_words_all = len(all_words)
    total_words_content = len(content_words)
    unique_word_pct = (len(set(all_words)) / total_words_all * 100) if total_words_all else 0
    action_verb_density = (len(action_verbs) / total_words_all * 100) if total_words_all else 0

    # Common words 
    common_words = Counter(content_words).most_common(8)

    # Top action verbs
    top_action_verbs = Counter(action_verbs).most_common(8)

    # Simple feedback rules
    feedback = []
    if avg_sentence_length > 22:
        feedback.append("Your sentences are long, opt for tighter bullet-style phrasing.")
    if bullet_count < 3 and len(lines) > 6:
        feedback.append("Consider using more bullet points for readability.")
    if action_verb_density < 3:
        feedback.append("Action verbs are a bit low, make sure to start bullets with strong verbs (built, led, shipped, improved).")
    if len(weak_verb_hits) >= 3:
        feedback.append("You’re using several weak verbs. Swap for stronger verbs where possible.")
    if flesch < 20:
        feedback.append("Readability is very dense, try to shorten sentences and reduce filler words.")
    if not feedback:
        feedback.append("Overall structure looks solid, but refine with stronger verbs + impact metrics (%, $, time saved).")

    return {
        # Raw counts
        "Total Words (All)": total_words_all,
        "Total Words (Content)": total_words_content,
        "Total Sentences": len(sentences),
        "Average Sentence Length": round(avg_sentence_length, 2),

        # Readability
        "Readability (Flesch)": round(flesch, 2),
        "Grade Level": round(grade, 2),

        # Signals
        "Unique Word %": round(unique_word_pct, 2),
        "Action Verb Density %": round(action_verb_density, 2),
        "Bullet Count": bullet_count,
        "Weak Verb Hits": [(v, weak_verb_replacements.get(v,[])) for v in set(action_verbs) if v in weak_verbs_set],

        # Lists
        "Most Common Words": common_words,
        "Named Entities": entities,
        "Top Action Verbs": top_action_verbs,

        # Feedback
        "Feedback": feedback
    }
def match_job_description(resume_text: str, job_text: str) -> dict:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf[0], tfidf[1])[0][0] * 100

    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())
    missing = [w for w in job_words - resume_words if w.isalpha() and len(w) > 3]

    return {
        "Match Score": round(score, 2),
        "Missing Keywords": missing[:15]
    }

def calculate_score(report: dict) -> int:
    score = 0

    if report["Action Verb Density %"] >= 8:
        score += 25
    elif report["Action Verb Density %"] >= 5:
        score += 15
    elif report["Action Verb Density %"] >= 3:
        score += 10
    
    if report["Bullet Count"] >= 8:
        score += 20
    elif report["Bullet Count"] >= 5:
        score += 12
    elif report["Bullet Count"] >= 3:
        score += 6
    
    if report["Readability (Flesch)"] >= 40:
        score += 20
    elif report["Readability (Flesch)"] >= 25:
        score += 12
    elif report["Readability (Flesch)"] >= 15:
        score += 6
    
    if report["Unique Word %"] >= 70:
        score += 20
    elif report["Unique Word %"] >= 55:
        score += 12
    elif report["Unique Word %"] >= 40:
        score += 6
    
    weak_count = len(report["Weak Verb Hits"])
    if weak_count == 0:
        score += 15
    elif weak_count == 1:
        score += 10
    elif weak_count == 2:
        score += 5
    
    return min(score, 100)


st.set_page_config(page_title="Resume NLP Analyzer", page_icon="📄", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    .score-box {
        background: linear-gradient(135deg, #1e3a5f, #0d6efd);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .score-number {
        font-size: 4rem;
        font-weight: 800;
        color: white;
    }
    .score-label {
        font-size: 1.1rem;
        color: #a0c4ff;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Resume NLP Analyzer")
st.write("Paste your resume below and get quick NLP-based insights + readability + action-verb strength.")
tab1, tab2 = st.tabs(["Single Resume", "Compare Resumes"])

with tab1:
    upload = st.file_uploader("Upload Resume PDF (Optional)", type=["pdf"])
    text_input = st.text_area("Resume Text", height=320, placeholder="Paste resume text here.")

    if upload is not None:
        import fitz
        with fitz.open(stream=upload.read(), filetype="pdf") as pdf:
            text_input = "\n".join(page.get_text() for page in pdf)
        st.success("PDF has loaded successfully.")
    job_input = st.text_area("Job Description (optional)", height=200, placeholder="Paste your desired job description here to get a match score")
    if st.button("🔍 Analyze", use_container_width=True):
        if text_input.strip():
            report = analyze(text_input)
            
            resume_score = calculate_score(report)
            color = "#28a745" if resume_score >= 70 else "#ffc107" if resume_score >= 45 else "#dc3545"
            st.markdown(f"""
                <div class="score-box" style="background: linear-gradient(135deg, #1a1a2e, {color});">
                    <div class="score-number">{resume_score}<span style="font-size:2rem">/100</span></div>
                    <div class="score-label">Overall Resume Score</div>
                </div>
            """, unsafe_allow_html=True)
            st.progress(resume_score / 100)

            st.subheader("Results")

            # Metrics
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Words (All)", report["Total Words (All)"])
            c2.metric("Words (Content)", report["Total Words (Content)"])
            c3.metric("Sentences", report["Total Sentences"])
            c4.metric("Avg Sentence Len", report["Average Sentence Length"])
            c5.metric("Flesch", report["Readability (Flesch)"])
            c6.metric("Grade Level", report["Grade Level"])

            # Signals
            s1, s2, s3 = st.columns(3)
            s1.metric("Unique Word %", report["Unique Word %"])
            s2.metric("Action Verb Density %", report["Action Verb Density %"])
            s3.metric("Bullet Count", report["Bullet Count"])

            st.markdown("### Feedback")
            for tip in report["Feedback"]:
                st.write(f"💡 {tip}")

            if job_input.strip():
                st.markdown("### Job Description Match")
                match = match_job_description(text_input, job_input)
                col1, col2 = st.columns(2)
                col1.metric("Match Score", f"{match['Match Score']}%")
                st.progress(match["Match Score"] / 100)
                st.markdown("**Missing Keywords:**")
                if match["Missing Keywords"]:
                    st.write(", ".join(match["Missing Keywords"]))
                else:
                    st.write("None ✅")

            left, right = st.columns(2)

            with left:
                st.markdown("### Most Common Words (Content)")
                import plotly.express as px
                import pandas as pd
                word_df = pd.DataFrame(report["Most Common Words"], columns=["Word", "Count"])
                fig = px.bar(word_df, x="Count", y="Word", orientation="h")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Top Action Verbs")
                st.table([{"Verb": v, "Count": c} for v, c in report["Top Action Verbs"]])

                st.markdown("### Weak Verb Hits")
                weak_hits = report["Weak Verb Hits"]
                if weak_hits:
                    for verb, suggestions in weak_hits:
                        st.markdown(f"**{verb}** -> try instead: {', '.join(suggestions)}")
                else:
                    st.write("None detected ✅")

            with right:
                st.markdown("### Named Entities (Top 10)")
                st.table([{"Entity": t, "Label": l} for t, l in report["Named Entities"]])

                with st.expander("Show raw report JSON"):
                    st.json(report)

        else:
            st.warning("Paste text first to analyze.")
with tab2:
    st.markdown("### Compare Two Texts")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Resume 1**")
        text1 = st.text_area("Resume 1 Text", height=300, placeholder="Paste first resume here...", key="compare_resume1")

    with col2:
        st.markdown("**Resume 2**")
        text2 = st.text_area("Resume 2 Text", height=300, placeholder="Paste second resume here...", key="compare_resume2")

    if st.button("⚖️ Compare", use_container_width=True):
        if text1.strip() and text2.strip():
            report1 = analyze(text1)
            report2 = analyze(text2)
            score1 = calculate_score(report1)
            score2 = calculate_score(report2)

            st.markdown("### Results")

            r1, r2 = st.columns(2)

            with r1:
                st.markdown("#### Resume 1")
                st.metric("Overall Score", f"{score1} / 100")
                st.metric("Action Verb Density %", report1["Action Verb Density %"])
                st.metric("Bullet Count", report1["Bullet Count"])
                st.metric("Readability (Flesch)", report1["Readability (Flesch)"])
                st.metric("Unique Word %", report1["Unique Word %"])

            with r2:
                st.markdown("#### Resume 2")
                st.metric("Overall Score", f"{score2} / 100")
                st.metric("Action Verb Density %", report2["Action Verb Density %"])
                st.metric("Bullet Count", report2["Bullet Count"])
                st.metric("Readability (Flesch)", report2["Readability (Flesch)"])
                st.metric("Unique Word %", report2["Unique Word %"])

            st.markdown("### 🏆 Winner")
            if score1 > score2:
                st.success("Resume 1 is stronger.")
            elif score2 > score1:
                st.success("Resume 2 is stronger.")
            else:
                st.info("Its a tie.")
        else:
            st.warning("Paste both resumes to compare.")

            
            



