"""Task 1: Statistical Analysis of Natural Language - Zipf's Law and Vocabulary Coverage."""

from src.nlp_app.vocab_coverage import VocabCoverageAnalyzer
from src.nlp_app.zipf_analysis import ZipfAnalyzer
import streamlit as st
import yaml
from pathlib import Path
import sys

st.set_page_config(page_title="Task 1: Statistical analysis", layout="wide")
st.title("Task 1: Statistical Analysis of Natural Language")

# Load config
sys.path.insert(0, str(Path(__file__).parent.parent))
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Check if corpus is loaded
if not st.session_state.get('corpus_loaded', False):
    st.warning("Please load and process a corpus first in the Load Data page.")
    st.stop()

st.markdown("""
### Task Objective
Empirically study statistical properties of Polish language and verify Zipf's law on a large corpus.

This page covers:
- **Zipf's Law Verification**: Analyze word frequency distribution
- **Vocabulary Coverage**: Calculate words needed for 90% / 95% corpus understanding
- **Language Core**: Identify most important words
""")

# ============================================================================
# SECTION 1: ZIPF'S LAW ANALYSIS
# ============================================================================

st.markdown("---")
st.header("Section 1: Zipf's Law Verification")

st.markdown("""
Zipf's law states that the frequency of a word is inversely proportional to its rank:
- `frequency x rank ~ constant`
- In log-log plot: `log(frequency) ~ -log(rank) + constant`

Expected slope in log-log regression: approximately -1 for natural language.
""")

# Analysis options
use_lemma = st.checkbox("Use lemmas (recommended)", value=True,
                        help="Use lemmatized forms instead of raw tokens")

if st.button("Run Zipf Analysis", type="primary"):
    with st.spinner("Analyzing word frequencies..."):
        analyzer = ZipfAnalyzer(output_dir=config['output']['plot_dir'])

        # Analyze frequencies
        freq_df = analyzer.analyze_frequency(st.session_state.tokens_df, use_lemma=use_lemma)
        st.session_state.freq_df = freq_df

        # Verify Zipf's law
        zipf_results = analyzer.verify_zipf_law(freq_df, fit_range=config['zipf']['log_log_fit_range'])
        st.session_state.zipf_results = zipf_results

        st.success("Zipf analysis complete")

# Display Zipf results
if 'freq_df' in st.session_state:
    freq_df = st.session_state.freq_df
    zipf_results = st.session_state.zipf_results

    # Show Zipf metrics
    st.markdown("#### Zipf's Law Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Log-Log Slope", f"{zipf_results['slope']:.3f}")
    with col2:
        st.metric("Expected Slope", f"{zipf_results['expected_slope']:.3f}")
    with col3:
        deviation = zipf_results['deviation']
        st.metric("Deviation", f"{deviation:.3f}",
                  delta=f"{'Good' if deviation < 0.2 else 'Fair' if deviation < 0.4 else 'Poor'}")

    st.markdown(f"""
    **Interpretation**: 
    - Slope = {zipf_results['slope']:.3f} (expected approximately -1.0)
    - Deviation = {deviation:.3f}
    - {"Good fit: language follows Zipf's law" if deviation < 0.2 else "Moderate fit" if deviation < 0.4 else "Poor fit"}
    """)

    # Generate and display plots
    st.markdown("#### Zipf's Law Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Log-Log Plot: Rank vs Frequency")
        with st.spinner("Generating plot..."):
            analyzer = ZipfAnalyzer(output_dir=config['output']['plot_dir'])
            plot_path = analyzer.plot_zipf(freq_df)
            st.image(plot_path)

    with col2:
        st.markdown("##### Rank × Frequency Constancy")
        with st.spinner("Generating plot..."):
            plot_path = analyzer.plot_rank_times_freq(freq_df)
            st.image(plot_path)

    # Top words analysis
    with st.expander("View Top Frequent Words"):
        st.markdown("#### Top Frequent Words")

        top_n = st.slider("Number of words to display", 10, 200, 50)

        # Function words analysis
        st.markdown("##### Function Words in Top Ranks")
        top_words = analyzer.analyze_function_words(freq_df, top_n=top_n)

        function_count = top_words['is_function'].sum()
        function_pct = (function_count / len(top_words)) * 100

        st.info(f"**{function_count}** out of top **{top_n}** words ({function_pct:.1f}%) are function words "
                "(articles, prepositions, conjunctions, pronouns, etc.)")

        st.dataframe(
            top_words[['rank', 'word', 'freq', 'rank_times_freq']],
            height=300
        )

        # Content words in middle range
        st.markdown("##### Content Words in Middle Range")

        col1, col2 = st.columns(2)
        with col1:
            min_rank = st.number_input("Min Rank", value=500, min_value=1, max_value=10000)
        with col2:
            max_rank = st.number_input("Max Rank", value=5000, min_value=min_rank+1, max_value=20000)

        content_pos = analyzer.analyze_content_words(
            freq_df, rank_range=(min_rank, max_rank))

        st.markdown("**POS Distribution in Middle Range:**")
        st.bar_chart(content_pos.set_index('pos'))

        st.dataframe(content_pos)

# ============================================================================
# SECTION 2: VOCABULARY COVERAGE
# ============================================================================

if 'freq_df' in st.session_state:
    st.markdown("---")
    st.header("Section 2: Vocabulary Coverage Analysis")

    st.markdown("""
    How many words do you need to know to understand 90% / 95% of the corpus?
    This analysis helps determine the **minimum vocabulary size** needed for different coverage levels.
    """)

    # Coverage targets
    targets = st.multiselect(
        "Select coverage targets",
        options=[0.80, 0.85, 0.90, 0.95, 0.99],
        default=[0.90, 0.95],
        format_func=lambda x: f"{int(x*100)}%"
    )

    if st.button("Calculate Coverage", type="primary"):
        with st.spinner("Calculating vocabulary coverage..."):
            analyzer = VocabCoverageAnalyzer(output_dir=config['output']['csv_dir'])

            # Calculate coverage
            results, freq_df_with_coverage = analyzer.calculate_coverage(
                st.session_state.freq_df,
                targets=targets
            )

            st.session_state.coverage_results = results
            st.session_state.freq_df_with_coverage = freq_df_with_coverage

            st.success("Coverage calculated successfully")

    # Display coverage results
    if 'coverage_results' in st.session_state:
        results = st.session_state.coverage_results
        freq_df = st.session_state.freq_df_with_coverage

        # Summary metrics
        st.markdown("#### Coverage Summary")

        cols = st.columns(len(targets) + 1)

        with cols[0]:
            st.metric("Total Unique Words",
                      f"{results['total_unique_words']:,}")

        for idx, target in enumerate(targets, 1):
            if target in results['targets']:
                target_data = results['targets'][target]
                with cols[idx]:
                    st.metric(
                        f"{int(target*100)}% Coverage",
                        f"{target_data['words_needed']:,} words",
                        delta=f"{target_data['percentage_of_vocab']:.1f}% of vocab"
                    )

        # Detailed breakdown
        with st.expander("View Detailed Breakdown"):
            analyzer = VocabCoverageAnalyzer(output_dir=config['output']['csv_dir'])
            for target in targets:
                if target in results['targets']:
                    target_data = results['targets'][target]

                    st.markdown(f"##### {int(target*100)}% Coverage Details")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Words Needed", f"{target_data['words_needed']:,}")
                    with col2:
                        st.metric("Actual Coverage", f"{target_data['actual_coverage']*100:.2f}%")
                    with col3:
                        st.metric("% of Total Vocabulary", f"{target_data['percentage_of_vocab']:.2f}%")

                    st.info(f"""
                    To understand **{int(target*100)}%** of this corpus, you need to know:
                    - **{target_data['words_needed']:,}** words
                    - This is only **{target_data['percentage_of_vocab']:.2f}%** of all unique words in the corpus
                    - The remaining **{100-target*100:.0f}%** consists of rare words
                    """)

                    st.markdown("#### Export Word List")
                    st.markdown("Download the list of words required to understand 90% of the text.")
                    csv_data = analyzer.export_coverage_list(freq_df, target=target)

                    st.download_button(
                        label=f"Download {int(target*100)}% Coverage List (CSV)",
                        data=csv_data,
                        file_name=f"vocab_coverage_{int(target*100)}.csv",
                        mime="text/csv"
                    )

        # Visualization
        st.markdown("#### Coverage Curve")

        with st.spinner("Generating coverage plot..."):
            analyzer = VocabCoverageAnalyzer(output_dir=config['output']['csv_dir'])
            plot_path = analyzer.plot_coverage(freq_df)
            st.image(plot_path, use_column_width=True)

        st.markdown("""
        **How to read this chart:**
        - X-axis (log scale): Number of words you know
        - Y-axis: Percentage of corpus you can understand
        - The curve shows diminishing returns - learning common words gives high coverage quickly
        """)

        # Word lookup
        with st.expander("Word Lookup Tool"):
            st.markdown("#### Search for a Word")

            search_word = st.text_input("Enter a word to see its statistics", "")
            if search_word:
                word_data = freq_df[freq_df['word'].str.lower() == search_word.lower()]

                if len(word_data) > 0:
                    word_info = word_data.iloc[0]

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rank", f"{int(word_info['rank']):,}")
                    with col2:
                        st.metric("Frequency", f"{int(word_info['freq']):,}")
                    with col3:
                        st.metric("Coverage", f"{word_info['cumulative_share']*100:.2f}%")
                    with col4:
                        percentile = (1 - word_info['rank'] / len(freq_df)) * 100
                        st.metric("Top Percentile", f"{percentile:.1f}%")
                else:
                    st.warning(f"Word '{search_word}' not found in corpus.")

# ============================================================================
# SECTION 3: LANGUAGE CORE
# ============================================================================

st.markdown("---")
st.header("Section 3: Language Core")

st.markdown("""
**Language Core** definition:
- Words that have the most unique neighbors (collocations).
These words connect with many different words in the language.
""")

analyzer = ZipfAnalyzer(output_dir=config['output']['plot_dir'])
with st.spinner("Analyzing core words (unique neighbors)..."):
    # Analyze core words based on unique neighbors
    # We need tokens_df for this
    core_words = analyzer.analyze_core_words(st.session_state.tokens_df, use_lemma=use_lemma, top_n=100)

col1, col2 = st.columns(2)
with col1:
    st.metric("Identified Core Words", len(core_words))

st.markdown("##### List of Core Words (by unique neighbors)")
st.dataframe(
    core_words[['rank', 'word', 'unique_neighbors', 'freq']],
    use_container_width=True
)
