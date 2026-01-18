"""Main Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="NLP Analysis - Polish Language",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("NLP Analysis Application")

st.markdown("""
### Polish Language Processing: Three Core Tasks
Welcome to the NLP Analysis Application for Polish language.

This application implements three main tasks from TASK.md:

### Load Data
**Data Loading and Preprocessing**
- Load Polish text corpus
- Tokenization and lemmatization
- POS tagging and sentence segmentation

### Task 1: Statistical Analysis
**Empirical Study of Polish Language**
- Verify Zipf's Law on word frequency distribution
- Calculate vocabulary needed for 90% / 95% corpus coverage
- Identify function words and content words
- Analyze word frequency patterns and rankings

### Task 2: Syntax Generatio
**Generator for Polish Learners**
- Generate grammatically correct sentences
- Subject-Verb-Object structure
- Support for different tenses, moods, negation, questions
- 100 nouns, 100 adjectives, 100 verbs lexicon

### Task 3: Semantic Analysis
**Collocation Analysis and Meaning**
- Part A: Adjective + Noun collocations (bipartite graph)
- Part B: Verb + Object collocations
- Color-coded semantic validity assessment
- Identify language core (high-connectivity words)
- Semi-automatic noun annotation (50% goal)

### Getting Started

1. Use the sidebar to navigate between pages
2. Start with Load Data to load your corpus
3. Complete tasks in order: Task 1, Task 2, Task 3

### Requirements

- Polish text corpus (approximately 100,000 tokens)
- spaCy Polish model: `python -m spacy download pl_core_news_lg`
""")
