import streamlit as st
import spacy
import textstat
from collections import Counter
import re

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


def analyze(text: str) -> dict:
    doc = nlp(text)

    # Sentences
    text_for_sentences = " ".join(line.strip() for line in text.splitlines() if line.strip())
    doc_sentences = nlp(text_for_sentences)
    sentences = list(doc_sentences.sents)

    # "Content words" (alpha, non-stop)
    content_words = [t.text.lower() for t in doc if t.is_alpha and not t.is_stop]

    # All alphabetic words (includes stopwords) for some metrics
    all_words = [t.text.lower() for t in doc if t.is_alpha]

    # Average sentence length (words per sentence)
    avg_sentence_length = (
        sum(len(s.text.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )

    # Readability
    flesch = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)

    # Named entities (top 10)
    entities = [(ent.text.strip().replace("\n", " "), ent.label_) for ent in doc.ents][:10]

    # Action verbs (filtered)
    bad_verbs = {"resume"}  # spaCy sometimes tags "resume" as VERB; not helpful here
    action_verbs = [
        t.lemma_.lower() for t in doc
        if t.pos_ == "VERB"
        and t.is_alpha
        and not t.is_stop
        and t.lemma_.lower() not in bad_verbs
    ]

    # Weak verbs list (customizable)
    weak_verbs_set = {
        "help", "work", "learn", "do", "make", "use", "try", "assist", "handle", "support"
    }
    weak_verb_hits = [v for v in action_verbs if v in weak_verbs_set]

    # Bullet detection (lines that start with -, *, •)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    bullet_lines = [ln for ln in lines if re.match(r"^(\-|\*|•)\s+", ln)]
    bullet_count = len(bullet_lines)

    # Resume signal metrics
    total_words_all = len(all_words)
    total_words_content = len(content_words)
    unique_word_pct = (len(set(all_words)) / total_words_all * 100) if total_words_all else 0
    action_verb_density = (len(action_verbs) / total_words_all * 100) if total_words_all else 0

    # Common words (content words only)
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
        "Weak Verb Hits": Counter(weak_verb_hits).most_common(8),

        # Lists
        "Most Common Words": common_words,
        "Named Entities": entities,
        "Top Action Verbs": top_action_verbs,

        # Feedback
        "Feedback": feedback
    }


st.set_page_config(page_title="Resume NLP Analyzer", page_icon="📄", layout="wide")

st.title("📄 Resume NLP Analyzer")
st.write("Paste your resume below and get quick NLP-based insights + readability + action-verb strength.")

text_input = st.text_area("Resume Text", height=320, placeholder="Paste resume text here...")

if st.button("Analyze"):
    if text_input.strip():
        report = analyze(text_input)

        st.subheader("Results")

        # Metrics row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Words (All)", report["Total Words (All)"])
        c2.metric("Words (Content)", report["Total Words (Content)"])
        c3.metric("Sentences", report["Total Sentences"])
        c4.metric("Avg Sentence Len", report["Average Sentence Length"])
        c5.metric("Flesch", report["Readability (Flesch)"])
        c6.metric("Grade Level", report["Grade Level"])

        # Signals row
        s1, s2, s3 = st.columns(3)
        s1.metric("Unique Word %", report["Unique Word %"])
        s2.metric("Action Verb Density %", report["Action Verb Density %"])
        s3.metric("Bullet Count", report["Bullet Count"])

        st.markdown("### Feedback")
        for tip in report["Feedback"]:
            st.write(f"✅ {tip}")

        left, right = st.columns(2)

        with left:
            st.markdown("### Most Common Words (Content)")
            st.table([{"Word": w, "Count": c} for w, c in report["Most Common Words"]])

            st.markdown("### Top Action Verbs")
            st.table([{"Verb": v, "Count": c} for v, c in report["Top Action Verbs"]])

            st.markdown("### Weak Verb Hits")
            weak_hits = report["Weak Verb Hits"]
            if weak_hits:
                st.table([{"Verb": v, "Count": c} for v, c in weak_hits])
            else:
                st.write("None detected ✅")

        with right:
            st.markdown("### Named Entities (Top 10)")
            st.table([{"Entity": t, "Label": l} for t, l in report["Named Entities"]])

            with st.expander("Show raw report JSON"):
                st.json(report)

    else:
        st.warning("Paste text first to analyze.")
