"""Data loading page - Load and process Polish text corpus."""

from src.nlp_app.preprocess import TextPreprocessor
from src.nlp_app.corpus_loader import CorpusLoader
import streamlit as st
import pandas as pd
import yaml
from pathlib import Path
import sys

st.set_page_config(page_title="Load Data", layout="wide")
st.title("Load data: Corpus Loading and Processing")

# Load config
sys.path.insert(0, str(Path(__file__).parent.parent))
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize session state
if 'corpus_loaded' not in st.session_state:
    st.session_state.corpus_loaded = False
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'tokens_df' not in st.session_state:
    st.session_state.tokens_df = None
if 'sentences_df' not in st.session_state:
    st.session_state.sentences_df = None

st.markdown("### Load the corpus")

# Check for cached data
cache_dir: Path = Path(config['corpus']['cache_dir'])
# Parquet files
cached_tokens: Path = cache_dir / "tokens.parquet"
cached_sentences: Path = cache_dir / "sentences.parquet"
# Fallback to CSV
cached_tokens_csv: Path = cache_dir / "tokens.csv"
cached_sentences_csv: Path = cache_dir / "sentences.csv"

# Allow loading data from cache
has_cache: bool = (cached_tokens.exists() or cached_tokens_csv.exists()) and (cached_sentences.exists() or cached_sentences_csv.exists())
if has_cache:
    st.info("Found cached preprocessing results in data/cache.")
    if st.button("Load from Cache", type="primary"):
        with st.spinner("Loading cached data..."):
            try:
                # Load tokens
                if cached_tokens.exists():
                    st.session_state.tokens_df = pd.read_parquet(cached_tokens)
                else:
                    st.session_state.tokens_df = pd.read_csv(cached_tokens_csv)

                # Load sentences
                if cached_sentences.exists():
                    st.session_state.sentences_df = pd.read_parquet(cached_sentences)
                else:
                    st.session_state.sentences_df = pd.read_csv(cached_sentences_csv)

                st.session_state.corpus_loaded = True
                st.session_state.documents = []

                st.success("Loaded data from cache!")

            except Exception as e:
                st.error(f"Error loading cache: {e}")

# Corpus directory selection
data_dir = Path(config['corpus']['data_dir'])

col_limit, col_random, col_model = st.columns([2, 1, 2])
with col_limit:
    limit_files = st.slider("Limit number of files (0 for all)", 0, len(list(data_dir.glob("*"))),
                            100, help="Limit the number of files to load to save memory")
with col_random:
    st.write("")  # Spacer
    st.write("")  # Spacer
    random_sample = st.checkbox(
        "Random sample", value=True, help="Pick files randomly if limit is set")
with col_model:
    spacy_model = st.selectbox(
        "Spacy model",
        options=["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
        index=2,
        help="Pick the SpaCy model: sm (small), md (medium), lg (large/accurate)"
    )

if st.button("Load Corpus", type="primary"):
    with st.spinner("Loading corpus..."):
        try:
            # Load documents
            loader = CorpusLoader(
                data_dir=data_dir,
                file_extensions=config['corpus']['file_extensions'],
                encoding=config['corpus']['encoding']
            )
            documents = loader.load_files(
                limit=limit_files if limit_files > 0 else None,
                randomize=random_sample
            )

            if not documents:
                st.error("No documents found!")
            else:
                st.session_state.documents = documents

                # Get basic stats
                stats = loader.get_stats(documents)
                st.success(
                    f"Successfully loaded {stats['num_documents']} documents")

                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", stats['num_documents'])
                with col2:
                    st.metric("Total Words", f"{stats['total_words']:,}")
                with col3:
                    st.metric("Avg Words/Doc",
                              f"{stats['avg_words_per_doc']:.0f}")

        except Exception as e:
            st.error(f"Error loading corpus: {str(e)}")

# Process corpus
if st.session_state.documents and st.button("Process Corpus (Tokenization, Lemmatization, POS)", type="primary"):
    with st.spinner("Processing corpus... This may take a minute."):
        try:
            preprocessor = TextPreprocessor(
                model_name=spacy_model,
                min_token_length=config['preprocessing']['min_token_length']
            )

            progress_bar = st.progress(0, text="Starting processing...")

            def update_token_progress(p):
                progress_bar.progress(
                    p * 0.5, text=f"Tokenizing documents... {int(p*100)}%")

            def update_sentence_progress(p):
                progress_bar.progress(
                    0.5 + (p * 0.5), text=f"Extracting sentences... {int(p*100)}%")

            # Process documents
            tokens_df = preprocessor.process_documents(st.session_state.documents, progress_callback=update_token_progress)
            sentences_df = preprocessor.get_sentences(st.session_state.documents, progress_callback=update_sentence_progress)

            tokens_df.to_parquet(cached_tokens)
            tokens_df.to_csv(cached_tokens_csv)
            sentences_df.to_parquet(cached_sentences)
            sentences_df.to_csv(cached_sentences_csv)

            progress_bar.progress(1.0, text="Processing complete!")

            st.session_state.tokens_df = tokens_df
            st.session_state.sentences_df = sentences_df
            st.session_state.corpus_loaded = True

            st.success(f"Successfully processed {len(tokens_df)} tokens in {len(sentences_df)} sentences")

            # Display sample
            st.markdown("### Sample processed tokens")
            st.dataframe(tokens_df.head(100), height=300)
            st.markdown("### Sample processed sentences")
            st.dataframe(sentences_df.head(100), height=300)

        except Exception as e:
            st.error(f"Error processing corpus: {str(e)}")
            st.info(
                f"Make sure spaCy model is installed: python -m spacy download {spacy_model}")

# Show corpus status
if st.session_state.corpus_loaded and st.session_state.tokens_df is not None and st.session_state.sentences_df is not None:
    st.success(
        "Corpus is loaded and processed. You can now use other analysis pages.")

    # Show stats
    st.markdown("### Corpus Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.documents:
            num_docs = len(st.session_state.documents)
        elif st.session_state.tokens_df is not None:
            num_docs = st.session_state.tokens_df['filename'].nunique()
        else:
            num_docs = 0
        st.metric("Documents", num_docs)
    with col2:
        st.metric("Sentences", len(st.session_state.sentences_df))
    with col3:
        st.metric("Tokens", len(st.session_state.tokens_df))
    with col4:
        unique_lemmas = st.session_state.tokens_df['lemma'].nunique()
        st.metric("Unique Lemmas", unique_lemmas)

    # POS distribution
    st.markdown("### POS Tag Distribution")
    pos_counts = st.session_state.tokens_df['pos'].value_counts()
    st.bar_chart(pos_counts)
else:
    st.info("Load and process a corpus to begin analysis.")
