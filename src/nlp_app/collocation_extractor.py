"""Module for extracting collocations (Adj-Noun, Verb-Noun) from text."""
# Done

import spacy
import pandas as pd
from typing import Tuple, Optional


class CollocationExtractor:
    """Extracts Adjective-Noun and Verb-Noun collocations using spaCy dependency parsing."""

    def __init__(self, model_name: Optional[str] = 'pl_core_news_lg'):
        """
        Initialize extractor with spaCy model.

        Args:
            model_name: Name of spaCy model to use. If None, spacy will not be loaded.
        """
        if model_name:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"Model {model_name} not found. Please install it using `python -m spacy download {model_name}`")
                raise
        else:
            self.nlp = None

    def process_dataframe(self, df: pd.DataFrame, use_lemma: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process preprocessed dataframe and extract collocations.

        Args:
            df: DataFrame with columns: lemma, pos, dep, head_lemma, head_pos

        Returns:
            Tuple of (adj_noun_df, verb_noun_df)
        """
        # Ensure columns exist
        target = 'lemma' if use_lemma else 'token'
        target_head = 'head_lemma' if use_lemma else 'head_token'
        required_cols = [target, 'pos', 'dep', target_head, 'head_pos']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"DataFrame missing required columns: {missing}")

        # Adjective-Noun connections
        # Case 1: Adjective modifies Noun (amod)
        adj_mask = (df['pos'] == 'ADJ') & (df['dep'] == 'amod') & (df['head_pos'] == 'NOUN')
        adj_df = df[adj_mask].copy()

        # Filter non-alpha
        # Assuming lemma / token and head_lemma / head_token are strings. If they are categories, we might need to convert or check categories.
        if isinstance(adj_df[target].dtype, pd.CategoricalDtype):
            adj_df[target] = adj_df[target].astype(str)
        if isinstance(adj_df[target_head].dtype, pd.CategoricalDtype):
            adj_df[target_head] = adj_df[target_head].astype(str)
        adj_df = adj_df[adj_df[target].str.isalpha() & adj_df[target_head].str.isalpha()]

        # Count
        adj_noun_counts = adj_df.groupby([target, target_head]).size().reset_index(name='count')
        adj_noun_counts.columns = ['adjective', 'noun', 'count']

        # Verb-Noun connections
        # Case 2: Noun is subject or object of Verb
        verb_mask = (df['pos'] == 'NOUN') & (df['head_pos'] == 'VERB') & (df['dep'].isin(['nsubj', 'obj', 'iobj', 'obl']))
        verb_df = df[verb_mask].copy()

        if isinstance(verb_df[target].dtype, pd.CategoricalDtype):
            verb_df[target] = verb_df[target].astype(str)
        if isinstance(verb_df[target_head].dtype, pd.CategoricalDtype):
            verb_df[target_head] = verb_df[target_head].astype(str)

        # Filter non-alpha
        verb_df = verb_df[verb_df[target].str.isalpha() & verb_df[target_head].str.isalpha()]

        # Count
        verb_noun_counts = verb_df.groupby([target_head, target]).size().reset_index(name='count')
        verb_noun_counts.columns = ['verb', 'noun', 'count']

        return adj_noun_counts, verb_noun_counts
