"""
Embedding model operations for text encoding
Handles loading and using sentence transformers for semantic search
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config import EMBEDDING_MODEL_NAME, MAX_SEQUENCE_LENGTH


class EmbeddingModel:
    """Manages sentence transformer embeddings for semantic search"""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding model

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        print(f"🔄 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = MAX_SEQUENCE_LENGTH
        print(f"✅ Embedding model loaded (dimension: {self.get_dimension()})")

    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True,
               show_progress: bool = False, batch_size: int = 64) -> np.ndarray:
        """
        Encode texts into embeddings

        Args:
            texts: List of text strings to encode
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings

    def encode_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Encode a single query string

        Args:
            query: Query string to encode
            normalize: Whether to normalize the embedding

        Returns:
            numpy array embedding vector
        """
        embedding = self.model.encode(
            [query],
            normalize_embeddings=normalize
        )[0].astype('float32')
        return embedding
