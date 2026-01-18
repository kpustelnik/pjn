"""Corpus loader module for loading text files from a directory."""
# Done

from pathlib import Path
from typing import List, Tuple, Optional
import regex as re

class CorpusLoader:
    """Load and normalize text files from a directory."""

    def __init__(self, data_dir: Path, file_extensions: Optional[List[str]] = None, encoding: str = "utf-8"):
        """
        Initialize corpus loader.

        Args:
            data_dir: Path to directory containing text files
            file_extensions: List of file extensions to load (e.g., ['.txt', '.md'])
            encoding: Text encoding (default: utf-8)
        """
        self.data_dir = data_dir
        self.file_extensions = file_extensions or ['.txt', '.md']
        self.encoding = encoding

    def list_files(self, limit: Optional[int] = None, randomize: bool = False) -> List[Path]:
        """
        List text files from the directory without loading content.

        Args:
            limit: Maximum number of files to list
            randomize: Whether to pick files randomly if limit is set

        Returns:
            List of file paths
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        # Get all matching files first
        files = []
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix in self.file_extensions:
                files.append(file_path)

        # Apply limit
        if limit and limit > 0 and limit < len(files):
            if randomize:
                import random
                files = random.sample(files, limit)
            else:
                files = files[:limit]

        return files

    def load_files(self, limit: Optional[int] = None, randomize: bool = False) -> List[Tuple[str, str]]:
        """
        Load text files from the directory.

        Args:
            limit: Maximum number of files to load
            randomize: Whether to pick files randomly if limit is set

        Returns:
            List of tuples (filename, content)
        """
        documents = []

        # Get all matching files first
        files = self.list_files(limit=limit, randomize=randomize)

        for file_path in files:
            try:
                content = self._load_file(file_path)
                documents.append((file_path.name, content))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")

        return documents

    def _load_file(self, file_path: Path) -> str:
        """
        Load and normalize a single file.

        Args:
            file_path: Path to the file

        Returns:
            Normalized text content
        """
        with open(file_path, 'r', encoding=self.encoding, errors='ignore') as f:
            content = f.read()

        # Add spaces around words to separate them from special characters
        # This ensures tokens like "-word" or "word-" become " word " 
        content = re.sub(r'((\p{L}|\p{M}|\-)+)', r' \1 ', content)
        
        # Normalize whitespace (collapse multiple spaces into one)
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        return content

    def get_stats(self, documents: List[Tuple[str, str]]) -> dict:
        """
        Get basic statistics about loaded documents.

        Args:
            documents: List of (filename, content) tuples

        Returns:
            Dictionary with statistics
        """
        total_chars = sum(len(content) for _, content in documents)
        total_words = sum(len(content.split()) for _, content in documents)

        return {
            'num_documents': len(documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_words_per_doc': total_words / len(documents) if documents else 0
        }
