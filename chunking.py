from typing import Dict, Any 


class TextChunk:
    """Represents a chunk of text with metadata."""
    
    def __init__(self, text: str, chunk_id: int, metadata: Dict[str, Any] = None):
        self.text = text
        self.chunk_id = chunk_id
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return f"TextChunk(id={self.chunk_id}, length={len(self.text)})"