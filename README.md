# 📄 Resume NLP Analyzer

A web-based NLP tool that analyzes resume text for readability, action verb strength, keyword frequency, and named entities.

🔗 **Live Demo:** [resume-nlp-analyzer.streamlit.app](https://resume-nlp-analyzer.streamlit.app)
---
## Features

- **Overall Resume Score** — scores your resume out of 100 based on readability, action verb density, bullet usage, and vocabulary
- **Readability Analysis** — Flesch Reading Ease and Flesch-Kincaid Grade Level scores
- **Action Verb Detection** — identifies strong action verbs and flags weak ones with replacement suggestions
- **Keyword Frequency** — visual bar chart of the most common content words
- **Named Entity Recognition** — extracts names, companies, dates, and locations using spaCy
- **Job Description Matcher** — paste a job description to get a match score and list of missing keywords
- **PDF Upload** — upload a resume PDF directly instead of pasting text
- **Compare Resumes** — side-by-side comparison of two resumes with a winner declaration
- **Export Report** — download your full analysis as a PDF
---
## Tech Stack
- Python
- spaCy
- textstat
- scikit-learn
- Streamlit
- Plotly
- PyMuPDF (fitz)
- fpdf2
---
## Run Locally
```bash
git clone https://github.com/vgoradia/resume-nlp-analyzer.git
cd resume-nlp-analyzer
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
```
---
## Project Structure
```
resume-nlp-analyzer/
├── app.py              # Main Streamlit app
├── main.py             # Original terminal version
├── requirements.txt    # Dependencies
├── sample_resume.txt   # Sample resume for testing
└── README.md
```
---

Built by Veer Goradia
