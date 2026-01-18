"""Vocabulary coverage analysis module."""
# Done

import pandas as pd
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

class VocabCoverageAnalyzer:
    """Analyze vocabulary coverage (e.g., words needed for 90% / 95% coverage)."""
    
    def __init__(self, output_dir: str = "./output/csv"):
        """
        Initialize vocabulary coverage analyzer.
        
        Args:
            output_dir: Directory for saving CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_coverage(self, freq_df: pd.DataFrame, targets: Optional[List[float]] = None) -> tuple[dict, pd.DataFrame]:
        """
        Calculate vocabulary needed for different coverage targets.
        
        Args:
            freq_df: DataFrame with rank, word, freq columns
            targets: List of target coverage percentages (e.g., [0.90, 0.95])
            
        Returns:
            Dictionary with coverage statistics
        """
        if targets is None:
            targets = [0.90, 0.95]

        # Filter out punctuation tokens if present
        if 'is_punct' in freq_df.columns:
            freq_df = freq_df[~freq_df['is_punct']].copy()
        
        # Calculate cumulative frequency
        total_tokens = freq_df['freq'].sum()
        freq_df = freq_df.copy()
        freq_df['cumulative_freq'] = freq_df['freq'].cumsum()
        freq_df['cumulative_share'] = freq_df['cumulative_freq'] / total_tokens
        
        results = {
            'total_tokens': int(total_tokens),
            'total_unique_words': len(freq_df),
            'targets': {}
        }
        
        for target in targets:
            # Find minimum N where cumulative_share >= target
            coverage_words = freq_df[freq_df['cumulative_share'] >= target]
            
            if len(coverage_words) > 0:
                n_words = coverage_words.iloc[0]['rank']
                actual_coverage = coverage_words.iloc[0]['cumulative_share']
                
                results['targets'][target] = {
                    'words_needed': int(n_words),
                    'actual_coverage': float(actual_coverage),
                    'percentage_of_vocab': float(n_words / len(freq_df) * 100)
                }
        
        return results, freq_df
    
    def export_coverage_list(self, freq_df: pd.DataFrame, target: float = 0.90, filename: Optional[str] = None) -> bytes:
        """
        Export list of words covering target percentage.
        
        Args:
            freq_df: DataFrame with cumulative_share column
            target: Target coverage (e.g., 0.90 for 90%)
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        # Get words up to target coverage
        coverage_words = freq_df[freq_df['cumulative_share'] <= target].copy()
        
        # Add one more word to exceed target
        if len(coverage_words) < len(freq_df):
            next_word = freq_df.iloc[len(coverage_words):len(coverage_words)+1]
            coverage_words = pd.concat([coverage_words, next_word])
        
        # Select and rename columns
        export_df = coverage_words[['rank', 'word', 'freq', 'cumulative_share']].copy()
        export_df.columns = ['rank', 'lemma', 'freq', 'cumulative_share']
        
        # Save to CSV
        return export_df.to_csv(index=False).encode('utf-8')
    
    def plot_coverage(self, freq_df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plot cumulative coverage curve.
        
        Args:
            freq_df: DataFrame with cumulative_share column
            save_path: Path to save plot (optional)
            
        Returns:
            Path to saved plot
        """
        
        if save_path is None:
            save_path = self.output_dir.parent / "plots" / "vocab_coverage.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot on log scale for x-axis to see details
        plt.semilogx(freq_df['rank'], freq_df['cumulative_share'] * 100, 'b-', linewidth=2)
        
        # Add horizontal lines for common targets
        for target in [50, 75, 90, 95]:
            plt.axhline(y=target, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            plt.text(1, target + 1, f'{target}%', fontsize=9, color='gray')
        
        plt.xlabel('Number of Words (log scale)')
        plt.ylabel('Corpus Coverage (%)')
        plt.title('Vocabulary Coverage: Percentage of Corpus vs Number of Words')
        plt.grid(True, alpha=0.3)
        plt.xlim(1, len(freq_df))
        plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        return str(save_path)
