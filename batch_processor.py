"""
Batch processor for handling multiple CVs.

Processes a folder of PDF files through the complete pipeline:
1. Extract text from PDF
2. Clean text
3. Extract structured data
4. Generate embeddings
5. Store in vector database
6. Index in metadata store
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import json
from tqdm import tqdm

from extract_text import extract_text_from_pdf
from text_cleaner import clean_and_structure_cv
from cv_extractor import extract_cv_data
from chunking import format_cv_for_search
from embedding import generate_embeddings
from vector_store import load_embeddings_to_store
from metadata_store import MetadataStore


def process_single_cv(
    pdf_path: str,
    output_dir: str = "./processed_cvs",
    use_llm_cleaning: bool = True,
    provider: str = None
) -> Dict[str, Any]:
    """
    Process a single CV through the complete pipeline.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Directory for output files
        use_llm_cleaning (bool): Whether to use LLM for text cleaning
        provider (str, optional): LLM provider
    
    Returns:
        Dict: Processing results
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    base_name = pdf_path.stem
    results = {
        "pdf_file": str(pdf_path),
        "candidate_name": None,
        "success": False,
        "error": None
    }
    
    try:
        print(f"\n{'='*70}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*70}")
        
        # Get API key for rotation (if using Gemini)
        from config import Config
        api_keys = Config.get_gemini_api_keys()
        current_api_key = None
        
        if api_keys and Config.LLM_PROVIDER == "gemini":
            # Rotate through keys based on file index or use modulo
            key_index = hash(str(pdf_path)) % len(api_keys)
            current_api_key = api_keys[key_index]
            print(f"🔑 Using API key #{key_index + 1} of {len(api_keys)}")
        
        # Step 1: Extract text
        print("1️⃣  Extracting text from PDF...")
        raw_text = extract_text_from_pdf(str(pdf_path))
        
        # Step 2: Clean text
        print("2️⃣  Cleaning text...")
        cleaned_result = clean_and_structure_cv(raw_text, provider=provider, api_key=current_api_key)
        cleaned_text = cleaned_result['cleaned_text']
        
        # Save cleaned text
        cleaned_file = output_dir / f"{base_name}_cleaned.txt"
        with open(cleaned_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Step 3: Extract structured data
        print("3️⃣  Extracting structured data...")
        cv_data = extract_cv_data(cleaned_text, source_file=str(pdf_path), provider=provider, api_key=current_api_key)
        results["candidate_name"] = cv_data.candidate_name
        
        # Save extracted data
        extracted_file = output_dir / f"{base_name}_extracted.json"
        with open(extracted_file, 'w', encoding='utf-8') as f:
            f.write(cv_data.to_json())
        
        # Step 4: Format CV data for vector search
        print("4️⃣  Formatting CV data for search...")
        chunks = format_cv_for_search(
            cv_data=cv_data,
            source_file=str(pdf_path)
        )
        print(f"   ✓ Created comprehensive profile chunk")
        
        # Step 5: Generate embeddings
        print("5️⃣  Generating embeddings...")
        embedded_chunks = generate_embeddings(chunks, show_progress=False)
        
        # Save embeddings
        embeddings_file = output_dir / f"{base_name}_embeddings.json"
        embeddings_data = [
            {
                "chunk_id": ec.chunk.chunk_id,
                "text": ec.chunk.text,
                "embedding": ec.embedding.tolist() if hasattr(ec.embedding, 'tolist') else ec.embedding,
                "embedding_dim": ec.embedding_dim,
                "metadata": ec.chunk.metadata
            }
            for ec in embedded_chunks
        ]
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings_data, f, indent=2, ensure_ascii=False)
        
        # Step 6: Add to metadata store
        print("6️⃣  Adding to metadata store...")
        metadata_store = MetadataStore()
        candidate_id = metadata_store.add_candidate(cv_data)
        results["candidate_id"] = candidate_id
        
        print(f"\n✅ Successfully processed: {cv_data.candidate_name}")
        print(f"   Candidate ID: {candidate_id}")
        print(f"   Chunks: {len(embedded_chunks)}")
        print(f"   Skills: {len(cv_data.technical_skills)}")
        
        results["success"] = True
        results["num_chunks"] = len(embedded_chunks)
        results["num_skills"] = len(cv_data.technical_skills)
        
    except Exception as e:
        print(f"\n❌ Error processing {pdf_path.name}: {e}")
        results["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    return results


def process_cv_folder(
    folder_path: str,
    output_dir: str = "./processed_cvs",
    use_llm_cleaning: bool = True,
    provider: str = None,
    load_to_vector_store: bool = True
) -> Dict[str, Any]:
    """
    Process all PDFs in a folder.
    
    Args:
        folder_path (str): Path to folder containing PDFs
        output_dir (str): Directory for output files
        use_llm_cleaning (bool): Whether to use LLM for cleaning
        provider (str, optional): LLM provider
        load_to_vector_store (bool): Whether to load embeddings to ChromaDB
    
    Returns:
        Dict: Processing summary
    """
    folder_path = Path(folder_path)
    
    # Find all PDF files
    pdf_files = list(folder_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING: {len(pdf_files)} CVs")
    print(f"{'='*70}\n")
    
    results = []
    
    # Process each PDF
    for pdf_file in tqdm(pdf_files, desc="Processing CVs"):
        result = process_single_cv(
            str(pdf_file),
            output_dir=output_dir,
            use_llm_cleaning=use_llm_cleaning,
            provider=provider
        )
        results.append(result)
    
    # Load all embeddings to vector store
    if load_to_vector_store:
        print(f"\n{'='*70}")
        print("Loading embeddings to vector database...")
        print(f"{'='*70}\n")
        
        output_path = Path(output_dir)
        embedding_files = list(output_path.glob("*_embeddings.json"))
        
        for emb_file in embedding_files:
            try:
                load_embeddings_to_store(str(emb_file), clear_existing=False)
            except Exception as e:
                print(f"⚠️  Failed to load {emb_file.name}: {e}")
    
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process CVs")
    parser.add_argument('folder', help='Folder containing PDF files')
    parser.add_argument('--output', default='./processed_cvs', help='Output directory')
    parser.add_argument('--provider', choices=['gemini', 'claude'], help='LLM provider')
    parser.add_argument('--no-llm-cleaning', action='store_true', help='Skip LLM cleaning')
    parser.add_argument('--skip-vector-store', action='store_true', help='Skip loading to vector store')
    
    args = parser.parse_args()
    
    summary = process_cv_folder(
        args.folder,
        output_dir=args.output,
        use_llm_cleaning=not args.no_llm_cleaning,
        provider=args.provider,
        load_to_vector_store=not args.skip_vector_store
    )
    
    # Save summary
    summary_file = Path(args.output) / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Summary saved to: {summary_file}")
