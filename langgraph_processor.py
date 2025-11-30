"""
LangGraph-based CV processing pipeline.

This module replaces batch_processor.py with a stateful graph orchestration
that provides better error handling, checkpointing, and observability.

Pipeline Steps (as Graph Nodes):
1. extract_text_node - Extract raw text from PDF
2. clean_text_node - LLM-based text cleaning
3. extract_cv_node - LLM-based extract structured CV data
4. chunk_cv_node - Format CV as comprehensive chunk
5. generate_embeddings_node - Generate embeddings
6. store_data_node - Store in ChromaDB and metadata store
"""

import os
import json
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from tqdm import tqdm
import traceback

from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Import existing pipeline components
from extract_text import extract_text_from_pdf
from text_cleaner import clean_and_structure_cv
from cv_extractor import extract_cv_data, CVData
from chunking import format_cv_for_search, TextChunk
from embedding import generate_embeddings, EmbeddedChunk
from vector_store import load_embeddings_to_store
from metadata_store import MetadataStore
from config import Config


# ============================================================================
# State Schema
# ============================================================================

class CVProcessingState(TypedDict):
    """State schema for CV processing graph."""
    
    # Input
    pdf_path: str
    output_dir: str
    
    # Processing data
    raw_text: Optional[str]
    cleaned_text: Optional[str]
    cv_data: Optional[Dict[str, Any]]  # CVData as dict for serialization
    chunks: Optional[List[Dict[str, Any]]]  # TextChunk as dicts
    embedded_chunks: Optional[List[Dict[str, Any]]]  # EmbeddedChunk as dicts
    
    # Results
    candidate_id: Optional[str]
    candidate_name: Optional[str]
    
    # Metadata
    current_step: str
    api_key: Optional[str]
    error: Optional[str]
    success: bool


# ============================================================================
# Node Functions
# ============================================================================

def extract_text_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 1: Extract text from PDF.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with raw_text
    """
    try:
        print(f"1️⃣  Extracting text from PDF: {Path(state['pdf_path']).name}")
        
        raw_text = extract_text_from_pdf(state['pdf_path'])
        
        return {
            **state,
            "raw_text": raw_text,
            "current_step": "extract_text",
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in extract_text_node: {e}")
        return {
            **state,
            "current_step": "extract_text",
            "error": f"Text extraction failed: {str(e)}",
            "success": False
        }


def clean_text_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 2: Clean text using LLM.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with cleaned_text
    """
    try:
        print("2️⃣  Cleaning text...")
        
        if not state.get('raw_text'):
            raise ValueError("No raw text available")
        
        # Get API key for rotation
        api_key = state.get('api_key')
        
        cleaned_result = clean_and_structure_cv(
            state['raw_text'],
            provider=Config.LLM_PROVIDER,
            api_key=api_key
        )
        
        cleaned_text = cleaned_result['cleaned_text']
        
        # Save cleaned text
        output_dir = Path(state['output_dir'])
        output_dir.mkdir(exist_ok=True)
        base_name = Path(state['pdf_path']).stem
        
        cleaned_file = output_dir / f"{base_name}_cleaned.txt"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"   ✓ Saved cleaned text: {cleaned_file.name}")
        
        return {
            **state,
            "cleaned_text": cleaned_text,
            "current_step": "clean_text",
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in clean_text_node: {e}")
        return {
            **state,
            "current_step": "clean_text",
            "error": f"Text cleaning failed: {str(e)}",
            "success": False
        }


def extract_cv_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 3: Extract structured CV data.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with cv_data
    """
    try:
        print("3️⃣  Extracting structured data...")
        
        if not state.get('cleaned_text'):
            raise ValueError("No cleaned text available")
        
        # Get API key for rotation
        api_key = state.get('api_key')
        
        cv_data = extract_cv_data(
            state['cleaned_text'],
            source_file=state['pdf_path'],
            provider=Config.LLM_PROVIDER,
            api_key=api_key
        )
        
        # Save extracted data
        output_dir = Path(state['output_dir'])
        base_name = Path(state['pdf_path']).stem
        
        extracted_file = output_dir / f"{base_name}_extracted.json"
        with open(extracted_file, 'w', encoding='utf-8') as f:
            f.write(cv_data.to_json())
        
        print(f"   ✓ Extracted data for: {cv_data.candidate_name}")
        
        return {
            **state,
            "cv_data": cv_data.to_dict(),
            "candidate_name": cv_data.candidate_name,
            "current_step": "extract_cv",
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in extract_cv_node: {e}")
        return {
            **state,
            "current_step": "extract_cv",
            "error": f"CV extraction failed: {str(e)}",
            "success": False
        }


def chunk_cv_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 4: Format CV data as chunks for embedding.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with chunks
    """
    try:
        print("4️⃣  Formatting CV data for search...")
        
        if not state.get('cv_data'):
            raise ValueError("No CV data available")
        
        # Reconstruct CVData object from dict
        cv_data = CVData(**state['cv_data'])
        
        chunks = format_cv_for_search(
            cv_data=cv_data,
            source_file=state['pdf_path']
        )
        
        # Convert chunks to dicts for serialization
        chunks_dict = [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
        
        print(f"   ✓ Created comprehensive profile chunk")
        
        return {
            **state,
            "chunks": chunks_dict,
            "current_step": "chunk_cv",
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in chunk_cv_node: {e}")
        return {
            **state,
            "current_step": "chunk_cv",
            "error": f"Chunking failed: {str(e)}",
            "success": False
        }


def generate_embeddings_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 5: Generate embeddings for chunks.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with embedded_chunks
    """
    try:
        print("5️⃣  Generating embeddings...")
        
        if not state.get('chunks'):
            raise ValueError("No chunks available")
        
        # Reconstruct TextChunk objects from dicts
        chunks = [
            TextChunk(
                chunk_id=c['chunk_id'],
                text=c['text'],
                metadata=c['metadata']
            )
            for c in state['chunks']
        ]
        
        embedded_chunks = generate_embeddings(chunks, show_progress=False)
        
        # Convert to dicts for serialization
        embedded_chunks_dict = [
            {
                "chunk_id": ec.chunk.chunk_id,
                "text": ec.chunk.text,
                "embedding": ec.embedding.tolist() if hasattr(ec.embedding, 'tolist') else ec.embedding,
                "embedding_dim": ec.embedding_dim,
                "metadata": ec.chunk.metadata
            }
            for ec in embedded_chunks
        ]
        
        # Save embeddings
        output_dir = Path(state['output_dir'])
        base_name = Path(state['pdf_path']).stem
        
        embeddings_file = output_dir / f"{base_name}_embeddings.json"
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embedded_chunks_dict, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Generated {len(embedded_chunks)} embeddings")
        
        return {
            **state,
            "embedded_chunks": embedded_chunks_dict,
            "current_step": "generate_embeddings",
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in generate_embeddings_node: {e}")
        return {
            **state,
            "current_step": "generate_embeddings",
            "error": f"Embedding generation failed: {str(e)}",
            "success": False
        }


def store_data_node(state: CVProcessingState) -> CVProcessingState:
    """
    Node 6: Store embeddings and metadata.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with candidate_id and success flag
    """
    try:
        print("6️⃣  Storing data...")
        
        if not state.get('cv_data'):
            raise ValueError("No CV data available")
        
        # Add to metadata store
        cv_data = CVData(**state['cv_data'])
        metadata_store = MetadataStore()
        candidate_id = metadata_store.add_candidate(cv_data)
        
        print(f"   ✓ Added to metadata store: {candidate_id}")
        print(f"\n✅ Successfully processed: {cv_data.candidate_name}")
        print(f"   Candidate ID: {candidate_id}")
        print(f"   Skills: {len(cv_data.technical_skills)}")
        
        return {
            **state,
            "candidate_id": candidate_id,
            "current_step": "store_data",
            "success": True,
            "error": None
        }
        
    except Exception as e:
        print(f"❌ Error in store_data_node: {e}")
        return {
            **state,
            "current_step": "store_data",
            "error": f"Storage failed: {str(e)}",
            "success": False
        }


def error_handler_node(state: CVProcessingState) -> CVProcessingState:
    """
    Error handler node - called when processing fails.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with error information
    """
    print(f"\n❌ Processing failed at step: {state.get('current_step', 'unknown')}")
    print(f"   Error: {state.get('error', 'Unknown error')}")
    
    return {
        **state,
        "success": False
    }


# ============================================================================
# Conditional Routing
# ============================================================================

def should_continue(state: CVProcessingState) -> str:
    """
    Determine if processing should continue or handle error.
    
    Args:
        state: Current processing state
        
    Returns:
        Next node name or END
    """
    if state.get('error'):
        return "error_handler"
    return "continue"


# ============================================================================
# Graph Construction
# ============================================================================

def create_cv_processing_graph() -> StateGraph:
    """
    Create and compile the CV processing graph.
    
    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(CVProcessingState)
    
    # Add nodes
    workflow.add_node("extract_text", extract_text_node)
    workflow.add_node("clean_text", clean_text_node)
    workflow.add_node("extract_cv", extract_cv_node)
    workflow.add_node("chunk_cv", chunk_cv_node)
    workflow.add_node("generate_embeddings", generate_embeddings_node)
    workflow.add_node("store_data", store_data_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Define edges (linear flow with error handling)
    workflow.add_edge(START, "extract_text")
    
    # Each node checks for errors before continuing
    workflow.add_conditional_edges(
        "extract_text",
        should_continue,
        {
            "continue": "clean_text",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "clean_text",
        should_continue,
        {
            "continue": "extract_cv",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "extract_cv",
        should_continue,
        {
            "continue": "chunk_cv",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "chunk_cv",
        should_continue,
        {
            "continue": "generate_embeddings",
            "error_handler": "error_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_embeddings",
        should_continue,
        {
            "continue": "store_data",
            "error_handler": "error_handler"
        }
    )
    
    # Final nodes lead to END
    workflow.add_edge("store_data", END)
    workflow.add_edge("error_handler", END)
    
    # Compile without checkpointer to avoid KeyError: '__start__' in this version
    # memory = MemorySaver()
    graph = workflow.compile()
    
    return graph


# ============================================================================
# Processing Functions
# ============================================================================

def process_single_cv_with_graph(
    pdf_path: str,
    output_dir: str = "./processed_cvs",
    graph: Optional[StateGraph] = None
) -> Dict[str, Any]:
    """
    Process a single CV using the LangGraph pipeline.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        graph: Pre-compiled graph (optional, will create if not provided)
        
    Returns:
        Processing results
    """
    if graph is None:
        graph = create_cv_processing_graph()
    
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get API key for rotation
    api_keys = Config.get_gemini_api_keys()
    current_api_key = None
    
    if api_keys and Config.LLM_PROVIDER == "gemini":
        key_index = hash(str(pdf_path)) % len(api_keys)
        current_api_key = api_keys[key_index]
        print(f"🔑 Using API key #{key_index + 1} of {len(api_keys)}")
    
    # Initialize state
    initial_state: CVProcessingState = {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "raw_text": None,
        "cleaned_text": None,
        "cv_data": None,
        "chunks": None,
        "embedded_chunks": None,
        "candidate_id": None,
        "candidate_name": None,
        "current_step": "start",
        "api_key": current_api_key,
        "error": None,
        "success": False
    }
    
    print(f"\n{'='*70}")
    print(f"Processing: {pdf_path.name}")
    print(f"{'='*70}")
    
    try:
        # Run the graph (stateless)
        # config = {"configurable": {"thread_id": str(pdf_path)}}
        final_state = graph.invoke(initial_state)
        
        # Load embeddings to vector store if successful
        if final_state.get('success'):
            embeddings_file = output_dir / f"{pdf_path.stem}_embeddings.json"
            if embeddings_file.exists():
                try:
                    load_embeddings_to_store(str(embeddings_file), clear_existing=False)
                    print(f"   ✓ Loaded to vector store")
                except Exception as e:
                    print(f"   ⚠️  Failed to load to vector store: {e}")
        
        return {
            "pdf_file": str(pdf_path),
            "candidate_name": final_state.get('candidate_name'),
            "candidate_id": final_state.get('candidate_id'),
            "success": final_state.get('success', False),
            "error": final_state.get('error'),
            "current_step": final_state.get('current_step')
        }
        
    except Exception as e:
        print(f"\n❌ Graph execution failed: {e}")
        traceback.print_exc()
        return {
            "pdf_file": str(pdf_path),
            "candidate_name": None,
            "success": False,
            "error": str(e)
        }


def process_cv_folder_with_graph(
    folder_path: str,
    output_dir: str = "./processed_cvs"
) -> Dict[str, Any]:
    """
    Process all PDFs in a folder using LangGraph.
    
    Args:
        folder_path: Path to folder containing PDFs
        output_dir: Directory for output files
        
    Returns:
        Processing summary
    """
    folder_path = Path(folder_path)
    
    # Find all PDF files
    pdf_files = list(folder_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(pdf_files)} CVs (LangGraph)")
    print(f"{'='*70}\n")
    
    # Create graph once for reuse
    graph = create_cv_processing_graph()
    
    results = []
    
    # Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Processing CVs"):
        result = process_single_cv_with_graph(
            str(pdf_file),
            output_dir=output_dir,
            graph=graph
        )
        results.append(result)
    
    # Summary
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "candidates": [r["candidate_name"] for r in results if r["success"]]
    }
    
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Total: {summary['total']}")
    print(f"Success: {summary['success']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['success'] > 0:
        print(f"\nProcessed candidates:")
        for name in summary['candidates']:
            print(f"  ✓ {name}")
    
    return summary


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process CVs using LangGraph orchestration"
    )
    parser.add_argument(
        'input',
        help='PDF file or folder containing PDF files'
    )
    parser.add_argument(
        '--output',
        default='./processed_cvs',
        help='Output directory (default: ./processed_cvs)'
    )
    parser.add_argument(
        '--provider',
        choices=['gemini', 'claude'],
        help='LLM provider (overrides .env)'
    )
    
    args = parser.parse_args()
    
    # Override provider if specified
    if args.provider:
        Config.LLM_PROVIDER = args.provider
    
    # Validate configuration
    Config.validate()
    
    input_path = Path(args.input)
    
    # Determine if input is file or folder
    if input_path.is_file():
        # Process single file
        result = process_single_cv_with_graph(
            str(input_path),
            output_dir=args.output
        )
        
        if result['success']:
            print(f"\n✅ Processing complete!")
        else:
            print(f"\n❌ Processing failed: {result.get('error')}")
            exit(1)
            
    elif input_path.is_dir():
        # Process folder
        summary = process_cv_folder_with_graph(
            str(input_path),
            output_dir=args.output
        )
        
        # Save summary
        summary_file = Path(args.output) / "processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        if summary['failed'] > 0:
            exit(1)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        exit(1)
