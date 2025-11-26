"""
Vector store management using ChromaDB for persistent storage and retrieval.
"""

import chromadb
from chromadb.config import Settings
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import Config


def create_vector_store(collection_name: str = "pdf_chunks", persist_directory: str = None):
    """
    Create or get a ChromaDB collection for storing embeddings.
    
    Args:
        collection_name (str): Name of the collection
        persist_directory (str, optional): Directory for persistent storage
    
    Returns:
        chromadb.Collection: The ChromaDB collection
    """
    persist_directory = persist_directory or Config.VECTOR_DB_PATH
    
    # Create persistent client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    print(f"Vector store initialized at: {persist_directory}")
    print(f"Collection: {collection_name}")
    print(f"Total documents: {collection.count()}")
    
    return collection


def load_embeddings_to_store(
    json_path: str,
    collection_name: str = "pdf_chunks",
    persist_directory: str = None,
    clear_existing: bool = False
) -> chromadb.Collection:
    """
    Load embeddings from JSON file into ChromaDB.
    
    Args:
        json_path (str): Path to embeddings JSON file
        collection_name (str): Name of the collection
        persist_directory (str, optional): Directory for persistent storage
        clear_existing (bool): Whether to clear existing data
    
    Returns:
        chromadb.Collection: The populated collection
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading {len(data)} chunks from {json_path}...")
    
    # Create or get collection
    collection = create_vector_store(collection_name, persist_directory)
    
    # Clear existing data if requested
    if clear_existing and collection.count() > 0:
        print("Clearing existing data...")
        persist_directory = persist_directory or Config.VECTOR_DB_PATH
        client = chromadb.PersistentClient(path=persist_directory)
        client.delete_collection(collection_name)
        collection = create_vector_store(collection_name, persist_directory)
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in data:
        # Create unique chunk ID by including source filename
        source_file = chunk['metadata'].get('source_file', 'unknown')
        filename = Path(source_file).stem if source_file != 'unknown' else 'unknown'
        chunk_id = f"{filename}_chunk_{chunk['chunk_id']}"
        
        # Skip if already exists (unless clearing)
        if not clear_existing:
            try:
                existing = collection.get(ids=[chunk_id])
                if existing['ids']:
                    continue
            except:
                pass
        
        ids.append(chunk_id)
        embeddings.append(chunk['embedding'])
        documents.append(chunk['text'])
        metadatas.append(chunk['metadata'])
    
    # Add to collection in batches
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        
        collection.add(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        
        print(f"Added batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
    
    print(f"✓ Loaded {len(ids)} chunks into vector store")
    print(f"Total documents in collection: {collection.count()}")
    
    return collection


def search_similar(
    query: str,
    collection_name: str = "pdf_chunks",
    top_k: int = None,
    score_threshold: float = None,
    metadata_filter: Dict[str, Any] = None,
    persist_directory: str = None
) -> List[Dict[str, Any]]:
    """
    Search for similar chunks using semantic search.
    
    Args:
        query (str): Query text
        collection_name (str): Name of the collection
        top_k (int, optional): Number of results to return
        score_threshold (float, optional): Minimum similarity score
        metadata_filter (Dict, optional): Filter by metadata
        persist_directory (str, optional): Directory for persistent storage
    
    Returns:
        List[Dict]: List of results with text, metadata, and scores
    """
    from embedding import generate_embedding
    
    top_k = top_k or Config.RETRIEVAL_TOP_K
    score_threshold = score_threshold or Config.RETRIEVAL_SCORE_THRESHOLD
    persist_directory = persist_directory or Config.VECTOR_DB_PATH
    
    # Get collection
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_collection(collection_name)
    
    # Generate query embedding
    print("📥 Loading embedding model and generating query embedding...")
    query_embedding = generate_embedding(query)
    print("✓ Query embedding generated")
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=metadata_filter
    )
    
    # Format results
    formatted_results = []
    for i in range(len(results['ids'][0])):
        # Calculate similarity score (ChromaDB returns distances, convert to similarity)
        distance = results['distances'][0][i]
        similarity = 1 - distance  # Cosine distance to similarity
        
        # Apply threshold
        if similarity < score_threshold:
            continue
        
        formatted_results.append({
            'id': results['ids'][0][i],
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'score': similarity
        })
    
    return formatted_results


def get_collection_stats(collection_name: str = "pdf_chunks", persist_directory: str = None) -> Dict[str, Any]:
    """
    Get statistics about the vector store collection.
    
    Args:
        collection_name (str): Name of the collection
        persist_directory (str, optional): Directory for persistent storage
    
    Returns:
        Dict: Collection statistics
    """
    persist_directory = persist_directory or Config.VECTOR_DB_PATH
    
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        collection = client.get_collection(collection_name)
        
        # Get sample to analyze metadata
        sample = collection.peek(limit=10)
        
        # Extract unique source files
        source_files = set()
        pages = set()
        
        for metadata in sample['metadatas']:
            if 'source_file' in metadata:
                source_files.add(metadata['source_file'])
            if 'page_number' in metadata:
                pages.add(metadata['page_number'])
        
        return {
            'collection_name': collection_name,
            'total_chunks': collection.count(),
            'source_files': list(source_files),
            'num_sources': len(source_files),
            'pages': sorted(list(pages)),
            'num_pages': len(pages)
        }
    except Exception as e:
        return {
            'error': str(e),
            'collection_name': collection_name,
            'exists': False
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <embeddings.json>")
        print("\nExample:")
        print("  python vector_store.py CV_Javier_Cruz_Nov25_embeddings.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not Path(json_file).exists():
        print(f"Error: File not found: {json_file}")
        sys.exit(1)
    
    # Load embeddings into vector store
    collection = load_embeddings_to_store(json_file, clear_existing=True)
    
    # Show stats
    stats = get_collection_stats()
    print(f"\n{'='*60}")
    print("Vector Store Statistics:")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"{key}: {value}")
