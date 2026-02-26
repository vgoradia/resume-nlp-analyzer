import streamlit as st
import spacy
import textstat
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def analyze(text):
    doc = nlp(text)

    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    sentences = list(doc.sents)

    avg_sentence_length = (
        sum(len(s.text.split()) for s in sentences) / len(sentences)
        if sentences else 0
    )

    readability = textstat.flesch_reading_ease(text)
    entities = [(ent.text, ent.label_)for ent in doc.ents]
    action_verbs = [
        token.lemma_ for token in doc
        if token.pos_ == "VERB" and not token.is_stop
    ]

    return {
        "Total Words": len(words),
        "Total Sentences": len(sentences),
        "Average Sentences Length": round(avg_sentence_length, 2),
        "Readability Score": round(readability, 2),
        "Most Common Words": Counter(words).most_common(5),
        "Named Entities": entities[:10],
        "Top Action Verbs": Counter(action_verbs).most_common(5),
    }

st.title("Resume NLP Analyzer")
st.write("Paste your resume below, to analyze it.")

text_input = st.text_area("Resume Text", height=300)

if st.button("Analyze"):
    if text_input.strip():
        report = analyze(text_input)
        st.subheader("The Results")
        st.metric("Total Words", report["Total Words"])
        st.metric("Total Sentences", report["Total Sentences"])
        st.metric("Average Sentence Length", report["Average Sentence Length"])
        st.metric("Readability Score", report["Readability Score"])
        st.write("**Most Common Words:**", report["Most Common Words"])
        st.write("**Named Entities:**", report["Named Entities"])
        st.write("**Top Action Verbs:**", report["Top Action Verbs"])
    else:
        st.warning("Paste text first to analyze.")
        