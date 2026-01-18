"""Text preprocessing module using spaCy."""
# Done

import spacy
import pandas as pd
import sys
from typing import List, Tuple

class TextPreprocessor:
    """Process and tokenize Polish text using spaCy."""

    def __init__(self, model_name='pl_core_news_lg', min_token_length=1):
        """
        Initialize preprocessor with spaCy model.

        Args:
            model_name: Name of spaCy model to use
            min_token_length: Minimum token length to keep
        """
        self.nlp = spacy.load(model_name)
        self.min_token_length = min_token_length

    def process_documents(self, documents: List[Tuple[str, str]], progress_callback=None) -> pd.DataFrame:
        """
        Process documents and extract tokens.

        Args:
            documents: List of (filename, text) tuples
            progress_callback: Optional callback function that accepts a float between 0 and 1

        Returns:
            DataFrame with columns: filename, sentence_id, token, lemma, pos, dep, head_token, head_lemma, head_pos, is_alpha, is_stop, is_punct
        """
        batch_data = {
            'filename': [],
            'sentence_id': [],
            'token': [],
            'lemma': [],
            'pos': [],
            'dep': [],
            'head_token': [],
            'head_lemma': [],
            'head_pos': [],
            'is_alpha': [],
            'is_stop': [],
            'is_punct': []
        }

        # Calculate total docs
        total_docs = len(documents)

        # Generator to avoid holding all chunks in memory
        def input_generator():
            for filename, text in documents:
                yield text, filename

        # Use nlp.pipe for faster processing
        # Disable ner as we only need tagging, lemmatization and sentence segmentation
        self.nlp.max_length = max(self.nlp.max_length, max(len(text) for _, text in documents) + 100)
        for idx, (doc, filename) in enumerate(self.nlp.pipe(input_generator(), as_tuples=True, disable=["ner"], batch_size=4, n_process=8)):
            current_sent_id: int = 0
            for sent in doc.sents:
                for token in sent:
                    # Skip tokens that are too short
                    if len(token.text) < self.min_token_length:
                        continue
                    
                    batch_data['filename'].append(filename)
                    batch_data['sentence_id'].append(current_sent_id)
                    # Use sys.intern to deduplicate strings in memory
                    batch_data['token'].append(sys.intern(token.text.lower()))
                    batch_data['lemma'].append(sys.intern(token.lemma_.lower()))
                    batch_data['pos'].append(sys.intern(token.pos_))
                    batch_data['dep'].append(sys.intern(token.dep_))
                    batch_data['head_token'].append(sys.intern(token.head.text.lower()))
                    batch_data['head_lemma'].append(sys.intern(token.head.lemma_.lower()))
                    batch_data['head_pos'].append(sys.intern(token.head.pos_))
                    batch_data['is_alpha'].append(token.is_alpha)
                    batch_data['is_stop'].append(token.is_stop)
                    batch_data['is_punct'].append(token.is_punct)

                current_sent_id += 1

            if progress_callback:
                progress_callback((idx + 1) / total_docs)

        return pd.DataFrame(batch_data)

    def get_sentences(self, documents: List[Tuple[str, str]], progress_callback=None) -> pd.DataFrame:
        """
        Extract sentences from documents.

        Args:
            documents: List of (filename, text) tuples
            progress_callback: Optional callback function that accepts a float between 0 and 1

        Returns:
            DataFrame with columns: filename, sentence_id, sentence
        """
        # Use dict of lists for memory efficiency
        data = {
            'filename': [],
            'sentence_id': [],
            'sentence': []
        }

        total_docs = len(documents)

        # Generator to avoid holding all chunks in memory
        def input_generator():
            for filename, text in documents:
                yield text, filename

        # Use nlp.pipe for faster processing
        # Disable ner as we only need sentence segmentation
        self.nlp.max_length = max(self.nlp.max_length, max(len(text) for _, text in documents) + 100)
        for idx, (doc, filename) in enumerate(self.nlp.pipe(input_generator(), as_tuples=True, disable=["ner"], batch_size=4, n_process=8)):
            current_id: int = 0

            for sent in doc.sents:
                data['filename'].append(filename)
                data['sentence_id'].append(current_id)
                data['sentence'].append(sent.text.strip())
                current_id += 1

            if progress_callback:
                progress_callback((idx + 1) / total_docs)

        return pd.DataFrame(data)
