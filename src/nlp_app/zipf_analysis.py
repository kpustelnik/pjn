"""Zipf's law analysis module."""
# Done

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple

class ZipfAnalyzer:
    """Analyze word frequency distribution and verify Zipf's law."""

    def __init__(self, output_dir: str = "./output/plots"):
        """
        Initialize Zipf analyzer.

        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_frequency(self, tokens_df: pd.DataFrame, use_lemma: bool = True) -> pd.DataFrame:
        """
        Analyze word frequency and create ranking.

        Args:
            tokens_df: DataFrame with token/lemma data
            use_lemma: Whether to use lemma or token for frequency analysis

        Returns:
            DataFrame with columns: rank, word, freq, rank_times_freq, pos
        """
        word_col = 'lemma' if use_lemma else 'token'

        # Remove punctuation tokens
        tokens_df = tokens_df[~tokens_df['is_punct']].copy()

        # Count frequencies and get POS (most common POS for each word, sum all POS counts)
        pos_counts = tokens_df.groupby([word_col, 'pos'], observed=False).size().reset_index(name='count')
        # For each word, get the most common POS
        most_common_pos = pos_counts.sort_values(['count', 'pos'], ascending=[False, True]).drop_duplicates(subset=[word_col])
        # Sum all POS counts per word
        freq_counts = pos_counts.groupby(word_col, observed=False)['count'].sum().reset_index(name='freq')
        # Merge most common POS
        freq_counts = freq_counts.merge(most_common_pos[[word_col, 'pos']], on=word_col, how='left')

        # Sort by frequency descending
        freq_counts = freq_counts.sort_values('freq', ascending=False).reset_index(drop=True)

        # Add rank
        freq_counts['rank'] = range(1, len(freq_counts) + 1)

        # Calculate rank * freq
        freq_counts['rank_times_freq'] = freq_counts['rank'] * freq_counts['freq']

        # Rename column
        freq_counts = freq_counts.rename(columns={ word_col: 'word' })

        return freq_counts[['rank', 'word', 'freq', 'rank_times_freq', 'pos']]

    def verify_zipf_law(self, freq_df: pd.DataFrame, fit_range: int = 1000) -> dict:
        """
        Verify Zipf's law by fitting log-log relationship.

        Args:
            freq_df: DataFrame with rank and freq columns
            fit_range: Number of top words to use for fitting

        Returns:
            Dictionary with fit parameters
        """
        # Take top N for fitting
        fit_data = freq_df.head(fit_range)

        log_rank = np.log(fit_data['rank'])
        log_freq = np.log(fit_data['freq'])

        # Fit: log(freq) = a + b * log(rank)
        # Expected: b ≈ -1 for natural language
        coeffs = np.polyfit(log_rank, log_freq, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        return {
            'slope': slope,
            'intercept': intercept,
            'fit_range': fit_range,
            'expected_slope': -1.0,
            'deviation': abs(slope - (-1.0))
        }

    def plot_zipf(self, freq_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Create log-log plot of rank vs frequency.

        Args:
            freq_df: DataFrame with rank and freq columns
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = str(self.output_dir / "zipf_plot.png")

        plt.figure(figsize=(10, 6))
        plt.loglog(freq_df['rank'], freq_df['freq'], 'b.', alpha=0.5, markersize=3)
        plt.xlabel('Rank (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Zipf\'s Law: Rank vs Frequency (log-log scale)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return str(save_path)

    def plot_rank_times_freq(self, freq_df: pd.DataFrame, save_path: Optional[str] = None) -> str:
        """
        Plot rank * freq vs rank to show constancy.

        Args:
            freq_df: DataFrame with rank and rank_times_freq columns
            save_path: Path to save plot (optional)

        Returns:
            Path to saved plot
        """
        if save_path is None:
            save_path = str(self.output_dir / "rank_times_freq.png")

        plt.figure(figsize=(10, 6))
        plt.plot(freq_df['rank'].head(1000), freq_df['rank_times_freq'].head(1000), 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Rank')
        plt.ylabel('Rank × Frequency')
        plt.title('Zipf\'s Law Verification: Rank × Frequency (should be approximately constant)')
        plt.grid(True, alpha=0.3)

        # Add horizontal line for mean
        mean_val = freq_df['rank_times_freq'].head(100).mean()
        plt.axhline(y=mean_val, color='r', linestyle='--', label=f'Mean (top 100): {mean_val:.1f}', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        return str(save_path)

    def analyze_function_words(self, freq_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        """
        Analyze function words in top frequency.

        Args:
            freq_df: Frequency DataFrame
            top_n: Number of top words to analyze

        Returns:
            DataFrame with function word analysis
        """
        top_words = freq_df.head(top_n).copy()

        # Define function word POS tags
        function_pos = ['ADP', 'CCONJ', 'SCONJ', 'PART', 'PRON', 'DET', 'AUX']
        top_words['is_function'] = top_words['pos'].isin(function_pos)

        return top_words

    def analyze_content_words(self, freq_df: pd.DataFrame,
                              rank_range: Tuple[int, int] = (500, 5000)) -> pd.DataFrame:
        """
        Analyze content words in middle rank range.

        Args:
            freq_df: Frequency DataFrame
            rank_range: Tuple of (min_rank, max_rank)

        Returns:
            DataFrame with content word analysis
        """
        min_rank, max_rank = rank_range
        middle_words = freq_df[
            (freq_df['rank'] >= min_rank) &
            (freq_df['rank'] <= max_rank)
        ].copy()

        # Count POS distribution
        pos_dist = middle_words['pos'].value_counts()

        return pos_dist.reset_index(name='count').rename(columns={'index': 'pos'})

    def analyze_core_words(self, tokens_df: pd.DataFrame, use_lemma: bool = True, top_n: int = 100) -> pd.DataFrame:
        """
        Identify core words based on unique neighbor count.
        Optimized for large datasets using integer codes and vectorization.

        Args:
            tokens_df: DataFrame with token/lemma data
            use_lemma: Whether to use lemma or token
            top_n: Number of top words to return

        Returns:
            DataFrame with columns: rank, word, unique_neighbors, freq
        """
        word_col = 'lemma' if use_lemma else 'token'

        # Work with a subset of columns to save memory
        # We create a copy to avoid modifying the original dataframe
        df = tokens_df[~tokens_df['is_punct']][['filename', 'sentence_id', word_col]].copy()

        # Convert to category to save memory and speed up comparisons
        if df[word_col].dtype.name != 'category':
            df[word_col] = df[word_col].astype('category')

        # Get codes to work with integers
        codes = df[word_col].cat.codes
        categories = df[word_col].cat.categories

        # Prepare arrays for vectorized operations
        # Handle filename: use codes if categorical, otherwise values
        if isinstance(df['filename'].dtype, pd.CategoricalDtype):
            file_arr = df['filename'].cat.codes.values
        else:
            file_arr = df['filename'].values

        sent_arr = df['sentence_id'].values
        code_arr = codes.values

        # Ensure we have enough data
        n = len(df)
        if n < 2:
            return pd.DataFrame(columns=['rank', 'word', 'unique_neighbors', 'freq'])

        # Create pairs of adjacent words (u, v)
        # u = words at i, v = words at i+1
        u = code_arr[:-1]
        v = code_arr[1:]

        # Check boundaries (same sentence, same file)
        valid_mask = (sent_arr[:-1] == sent_arr[1:]) & \
                     (file_arr[:-1] == file_arr[1:]) & \
                     (u != -1) & (v != -1)

        # Filter valid pairs
        u = u[valid_mask]
        v = v[valid_mask]

        # Normalize pairs to (min, max) to identify unique undirected edges
        # This allows us to drop duplicates efficiently
        min_uv = np.minimum(u, v)
        max_uv = np.maximum(u, v)

        # Create DataFrame of unique edges
        # This drastically reduces the size from ~N_tokens to ~N_unique_collocations
        edges_df = pd.DataFrame({ 'u': min_uv, 'v': max_uv })
        unique_edges = edges_df.drop_duplicates()

        # Now count neighbors for each word
        # We need to consider the edge in both directions for the count
        all_directed = pd.concat([
            unique_edges.rename(columns={'u': 'word', 'v': 'neighbor'}),
            unique_edges.rename(columns={'v': 'word', 'u': 'neighbor'})
        ])

        # Count neighbors (simple count since edges are unique)
        neighbor_counts = all_directed.groupby('word')['neighbor'].count()

        # Map back to string words
        result = pd.DataFrame({
            'word': categories[neighbor_counts.index],
            'unique_neighbors': neighbor_counts.values
        })

        # Get frequencies for context
        # Use codes for speed and consistency
        valid_codes = code_arr[code_arr != -1]
        freq_counts = pd.Series(valid_codes).value_counts()

        freq_df = pd.DataFrame({
            'word': categories[freq_counts.index],
            'freq': freq_counts.values
        })

        # Calculate rank for freq_df (highest freq = rank 1)
        freq_df = freq_df.sort_values('freq', ascending=False).reset_index(drop=True)
        freq_df['rank'] = range(1, len(freq_df) + 1)

        # Merge results
        result = pd.merge(result, freq_df, on='word')

        # Sort by unique neighbors
        result = result.sort_values('unique_neighbors', ascending=False).head(top_n)

        return result
