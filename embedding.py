import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import time
from config import Config
from chunking import TextChunk

# Global variable to cache the local model
_local_model: Optional[SentenceTransformer] = None


class EmbeddedChunk:
    """Represents a text chunk with its embedding."""
    
    def __init__(self, chunk: TextChunk, embedding: List[float]):
        self.chunk = chunk
        self.embedding = embedding
    
    @property
    def embedding_dim(self) -> int:
        """Get the dimension of the embedding."""
        return len(self.embedding)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "metadata": self.chunk.metadata,
            "embedding": self.embedding,
            "embedding_dim": self.embedding_dim
        }
    
    def __repr__(self) -> str:
        return f"EmbeddedChunk(id={self.chunk.chunk_id}, dim={self.embedding_dim})"


def get_local_model(model_name: str = None) -> SentenceTransformer:
    """
    Get or initialize the local embedding model.
    
    Args:
        model_name (str, optional): Model name. Defaults to Config.LOCAL_EMBEDDING_MODEL
    
    Returns:
        SentenceTransformer: The loaded model
    """
    global _local_model
    model_name = model_name or Config.LOCAL_EMBEDDING_MODEL
    
    if _local_model is None:
        print(f"📥 Loading local embedding model: {model_name}...")
        _local_model = SentenceTransformer(model_name)
        print("✓ Model loaded")
    
    return _local_model


def generate_embedding(text: str, model_name: str = None) -> List[float]:
    """
    Generate embedding using local Sentence Transformer model.
    
    Args:
        text (str): Text to embed
        model_name (str, optional): Model name. Defaults to Config.LOCAL_EMBEDDING_MODEL
    
    Returns:
        List[float]: Embedding vector
    """
    model = get_local_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def generate_embeddings(
    chunks: List[TextChunk],
    model_name: str = None,
    show_progress: bool = True
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for multiple text chunks using local model.
    
    Args:
        chunks (List[TextChunk]): List of text chunks to embed
        model_name (str, optional): Model name. Defaults to Config.LOCAL_EMBEDDING_MODEL
        show_progress (bool): Whether to show progress
    
    Returns:
        List[EmbeddedChunk]: List of chunks with embeddings
    """
    if show_progress:
        print(f"🔧 Generating embeddings using local model...")
    
    local_model = get_local_model(model_name)
    texts = [chunk.text for chunk in chunks]
    
    # Encode all texts at once (much faster than one-by-one)
    embeddings = local_model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
    
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(EmbeddedChunk(chunk, embedding.tolist()))
    
    if show_progress:
        print(f"✓ Successfully embedded {len(embedded_chunks)} chunks")
    
    return embedded_chunks
