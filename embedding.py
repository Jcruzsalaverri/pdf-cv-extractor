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


def initialize_gemini_api(api_key: str = None) -> None:
    """
    Initialize the Gemini API with the provided API key.
    
    Args:
        api_key (str, optional): API key. If None, uses Config.GEMINI_API_KEY
    """
    api_key = api_key or Config.GEMINI_API_KEY
    if not api_key:
        raise ValueError("Gemini API key not provided")
    
    genai.configure(api_key=api_key)


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


def generate_local_embedding(text: str, model_name: str = None) -> List[float]:
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


def generate_gemini_embedding(
    text: str,
    model: str = None,
    task_type: str = "retrieval_document"
) -> List[float]:
    """
    Generate embedding for a single text using Gemini API.
    
    Args:
        text (str): Text to embed
        model (str, optional): Model name. Defaults to Config.EMBEDDING_MODEL
        task_type (str): Task type for embedding
    
    Returns:
        List[float]: Embedding vector
    """
    model = model or Config.EMBEDDING_MODEL
    result = genai.embed_content(
        model=model,
        content=text,
        task_type=task_type
    )
    return result['embedding']


def generate_embedding(
    text: str,
    model: str = None,
    provider: str = None
) -> List[float]:
    """
    Generate embedding using configured provider.
    
    Args:
        text (str): Text to embed
        model (str, optional): Model name
        provider (str, optional): "local" or "gemini". Defaults to Config.EMBEDDING_PROVIDER
    
    Returns:
        List[float]: Embedding vector
    """
    provider = provider or Config.EMBEDDING_PROVIDER
    
    if provider == "local":
        return generate_local_embedding(text, model)
    else:  # gemini
        return generate_gemini_embedding(text, model)


def generate_embeddings(
    chunks: List[TextChunk],
    model: str = None,
    batch_size: int = None,
    show_progress: bool = True,
    provider: str = None
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for multiple text chunks with batch processing.
    
    Args:
        chunks (List[TextChunk]): List of text chunks to embed
        model (str, optional): Model name
        batch_size (int, optional): Batch size. Defaults to Config.EMBEDDING_BATCH_SIZE
        show_progress (bool): Whether to show progress
        provider (str, optional): "local" or "gemini". Defaults to Config.EMBEDDING_PROVIDER
    
    Returns:
        List[EmbeddedChunk]: List of chunks with embeddings
    """
    provider = provider or Config.EMBEDDING_PROVIDER
    batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
    
    embedded_chunks = []
    total_chunks = len(chunks)
    
    # For local embeddings, we can process in larger batches efficiently
    if provider == "local":
        if show_progress:
            print(f"🔧 Generating embeddings using local model...")
        
        local_model = get_local_model(model)
        texts = [chunk.text for chunk in chunks]
        
        # Encode all texts at once (much faster)
        embeddings = local_model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
        
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunks.append(EmbeddedChunk(chunk, embedding.tolist()))
        
        if show_progress:
            print(f"✓ Successfully embedded {len(embedded_chunks)}/{total_chunks} chunks")
        
        return embedded_chunks
    
    # Gemini API processing (original batch logic)
    if show_progress:
        print(f"🔧 Generating embeddings using Gemini API...")
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        
        if show_progress:
            print(f"Processing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}...")
        
        for chunk in batch:
            try:
                embedding = generate_gemini_embedding(chunk.text, model=model)
                embedded_chunks.append(EmbeddedChunk(chunk, embedding))
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error embedding chunk {chunk.chunk_id}: {str(e)}")
                # Continue with next chunk instead of failing completely
                continue
    
    if show_progress:
        print(f"✓ Successfully embedded {len(embedded_chunks)}/{total_chunks} chunks")
    
    return embedded_chunks


def embed_chunks_with_retry(
    chunks: List[TextChunk],
    model: str = None,
    max_retries: int = 3,
    show_progress: bool = True,
    provider: str = None
) -> List[EmbeddedChunk]:
    """
    Generate embeddings with retry logic for failed chunks.
    
    Args:
        chunks (List[TextChunk]): List of text chunks to embed
        model (str, optional): Model name
        max_retries (int): Maximum number of retries for failed chunks
        show_progress (bool): Whether to show progress
        provider (str, optional): "local" or "gemini"
    
    Returns:
        List[EmbeddedChunk]: List of chunks with embeddings
    """
    provider = provider or Config.EMBEDDING_PROVIDER
    
    # For local embeddings, no retry needed (it's deterministic)
    if provider == "local":
        return generate_embeddings(chunks, model=model, show_progress=show_progress, provider=provider)
    
    # For Gemini API, use retry logic
    embedded_chunks = []
    failed_chunks = chunks.copy()
    
    for attempt in range(max_retries):
        if not failed_chunks:
            break
        
        if show_progress and attempt > 0:
            print(f"\nRetry attempt {attempt}/{max_retries} for {len(failed_chunks)} failed chunks...")
        
        current_embedded = generate_embeddings(
            failed_chunks,
            model=model,
            show_progress=show_progress,
            provider=provider
        )
        
        embedded_chunks.extend(current_embedded)
        
        # Identify chunks that still failed
        embedded_ids = {ec.chunk.chunk_id for ec in current_embedded}
        failed_chunks = [c for c in failed_chunks if c.chunk_id not in embedded_ids]
        
        if failed_chunks and attempt < max_retries - 1:
            time.sleep(2)  # Wait before retry
    
    if failed_chunks and show_progress:
        print(f"\n⚠️  Warning: {len(failed_chunks)} chunks failed after {max_retries} attempts")
    
    return embedded_chunks
