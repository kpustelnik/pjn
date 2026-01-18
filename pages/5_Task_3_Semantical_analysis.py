"""Task 3: Collocation Analysis (Adjective-Noun, Verb-Noun)."""

from src.nlp_app.collocation_extractor import CollocationExtractor
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import sys


st.set_page_config(page_title="Task 3: Collocations analysis", layout="wide")
st.title("Task 3: Collocations analysis")

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
Analyze which adjectives connect with which nouns, and which verbs connect with which nouns.
Visualize the connections as a bipartite graph.
""")

# Check for cached data
cache_dir = Path(config['corpus']['cache_dir'])
adj_noun_path = cache_dir / "adj_noun_collocations.parquet"
verb_noun_path = cache_dir / "verb_noun_collocations.parquet"

# Check if we have final merged files OR partial files
has_final = adj_noun_path.exists() and verb_noun_path.exists()

col_model, = st.columns(1)
with col_model:
    spacy_model = st.selectbox(
        "Spacy model",
        options=["pl_core_news_sm", "pl_core_news_md", "pl_core_news_lg"],
        index=2,
        help="Pick the SpaCy model: sm (small), md (medium), lg (large/accurate)"
    )

use_lemma = st.checkbox("Use lemmas (recommended)", value=True,
                        help="Use lemmatized forms instead of raw tokens")

if st.button("Prepare Collocation Data"):
    with st.spinner("Extracting collocations..."):
        extractor = CollocationExtractor(model_name=spacy_model)
        adj_noun_df, verb_noun_df = extractor.process_dataframe(st.session_state.tokens_df, use_lemma=use_lemma)
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.adj_noun_df = adj_noun_df
        st.session_state.verb_noun_df = verb_noun_df
        adj_noun_df.to_parquet(adj_noun_path)
        verb_noun_df.to_parquet(verb_noun_path)

        st.success("Extraction complete!")
        st.rerun()

if st.session_state.get('adj_noun_df') is not None and st.session_state.get('verb_noun_df') is not None:
    adj_noun_df = st.session_state.adj_noun_df
    verb_noun_df = st.session_state.verb_noun_df

    st.success(f"Loaded {len(adj_noun_df)} Adj-Noun pairs and {len(verb_noun_df)} Verb-Noun pairs.")

    # Analysis Type Selection
    analysis_type = st.radio("Select Analysis Type", [ "Adjective-Noun", "Verb-Noun" ])

    if analysis_type == "Adjective-Noun":
        df = adj_noun_df
        col1_name = "adjective"
        col2_name = "noun"
    else:
        df = verb_noun_df
        col1_name = "verb"
        col2_name = "noun"

    # Filter Top N
    top_n = st.slider("Top N", min_value=10, max_value=len(df), value=100, step=10)
    st.subheader(f"Top {top_n} {col1_name.capitalize()}s and {col2_name.capitalize()}s")

    # Get top N items for each column
    top_col1 = df.groupby(col1_name)['count'].sum().nlargest(top_n).index
    top_col2 = df.groupby(col2_name)['count'].sum().nlargest(top_n).index

    # Filter dataframe
    filtered_df = df[df[col1_name].isin(top_col1) & df[col2_name].isin(top_col2)]

    st.write(f"Filtered data contains {len(filtered_df)} connections.")

    # Display List
    st.subheader("Connection List")
    st.dataframe(filtered_df.sort_values('count', ascending=False))

    # Visualization
    st.subheader("Bipartite Graph Visualization")

    # Color scale logic
    def get_color(count):
        if count <= 1:
            return 'red'
        elif count == 2:
            return 'yellow'
        elif count > 10:
            return 'blue'
        else:  # 3 to 10
            return 'green'


    # Create graph
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(filtered_df[col1_name].unique(), bipartite=0)
    G.add_nodes_from(filtered_df[col2_name].unique(), bipartite=1)

    # Add edges with colors
    edges = []
    colors = []
    weights = []

    for _, row in filtered_df.iterrows():
        u = row[col1_name]
        v = row[col2_name]
        count = row['count']

        G.add_edge(u, v, weight=count)
        edges.append((u, v))
        colors.append(get_color(count))
        weights.append(count)

    # Draw
    fig, ax = plt.subplots(figsize=(12, 12))
    # Use bipartite layout to put nodes in two lines
    pos = nx.bipartite_layout(G, filtered_df[col1_name].unique())

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=filtered_df[col1_name].unique(), node_color='lightblue', node_size=100, label=col1_name)
    nx.draw_networkx_nodes(G, pos, nodelist=filtered_df[col2_name].unique(), node_color='lightgreen', node_size=100, label=col2_name)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, width=1, alpha=0.5)

    # Draw labels (optional, might be too crowded)
    if st.checkbox("Show Labels", value=True):
        nx.draw_networkx_labels(G, pos, font_size=8)
    st.pyplot(fig)

    # Detailed List for specific item
    st.subheader(f"Detailed List for a specific {col1_name}")
    selected_item = st.selectbox(f"Select {col1_name}", sorted(filtered_df[col1_name].unique()))

    if selected_item:
        item_connections = filtered_df[filtered_df[col1_name] == selected_item]
        item_connections.sort_values('count', ascending=False, inplace=True)
        st.write(f"Connections for **{selected_item}**:")

        # Format: noun[count]
        formatted_list = [
            f"{row[col2_name]}[{row['count']}]"
            for _, row in item_connections.iterrows()
        ]
        st.write(", ".join(formatted_list))
