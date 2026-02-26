import spacy
import textstat
from collections import Counter

nlp = spacy.load("en_core_web_sm")

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def analyze(text):
    doc = nlp(text)

    words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    sentences = list(doc.sents)

    avg_sentence_length = sum(len(s.text.split()) for s in sentences)/len(sentences) if sentences else 0
    readability = textstat.flesch_reading_ease(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    action_verbs = [
        token.lemma_ for token in doc
        if token.pos_ == "VERB" and not token.is_stop
    ]

    return {
        "Total Words": len(words),
        "Total Sentences": len(sentences),
        "Average Sentence Length": round(avg_sentence_length, 2),
        "Readability Score": round(readability, 2),
        "Most Common Words": Counter(words).most_common(5),
        "Named Entities": entities[:10],
        "Top Action Verbs": Counter(action_verbs).most_common(5),
    }
if __name__ == "__main__":
    text = load_text("sample_resume.txt")
    report = analyze(text)

    print("\n=== Resume NLP Analyzer Report === ")
    for key, value in report.items():
        print(f"{key}: {value}")

