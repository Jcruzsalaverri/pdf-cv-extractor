"""
Script to search CVs using semantic vector search.
"""

import sys
import time
from vector_store import search_similar
from metadata_store import MetadataStore

def search_candidates(query: str, top_k: int = 10):
    """
    Search for candidates using semantic vector search.
    
    Args:
        query (str): Search query (e.g., "Python developer with ML experience")
        top_k (int): Number of chunks to retrieve
    """
    start_time = time.time()
    
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 70)
    
    # 1. Perform Vector Search (Semantic)
    print("\n[1/3] Performing semantic vector search...")
    search_start = time.time()
    results = search_similar(query, top_k=top_k)
    search_time = time.time() - search_start
    print(f"✓ Vector search completed in {search_time:.2f}s - Found {len(results)} matching chunks")
    
    if not results:
        print("❌ No matching candidates found.")
        return

    # 2. Aggregate Results by Candidate
    print("\n[2/3] Aggregating results by candidate...")
    agg_start = time.time()
    candidates = {}
    
    for r in results:
        source_file = r['metadata'].get('source_file')
        candidate_name = r['metadata'].get('candidate_name', 'Unknown')
        
        if not source_file:
            continue
            
        # Initialize candidate if not seen
        if source_file not in candidates:
            candidates[source_file] = {
                "source_file": source_file,
                "candidate_name": candidate_name,
                "score": r['score'],  # Take best chunk score
                "best_chunk": r['text'],
                "matches": 1,
                "metadata": r['metadata']
            }
        else:
            candidates[source_file]["matches"] += 1
            # Keep highest score
            if r['score'] > candidates[source_file]["score"]:
                candidates[source_file]["score"] = r['score']
                candidates[source_file]["best_chunk"] = r['text']
    
    agg_time = time.time() - agg_start
    print(f"✓ Aggregated into {len(candidates)} candidates in {agg_time:.2f}s")

    # 3. Display Ranked Results
    print(f"\n[3/3] Ranking and displaying results...")
    total_time = time.time() - start_time
    print(f"\n✅ Found {len(candidates)} relevant candidates:\n")
    
    # Sort by score (descending)
    ranked_candidates = sorted(candidates.values(), key=lambda x: x['score'], reverse=True)
    
    for i, c in enumerate(ranked_candidates, 1):
        print(f"{i}. {c['candidate_name']}")
        print(f"   📄 Source: {c['source_file']}")
        print(f"   🎯 Match Score: {c['score']:.4f}")
        print(f"   📊 Matching Chunks: {c['matches']}")
        print(f"   💡 Best Match: \"{c['best_chunk'][:200]}...\"")
        print("=" * 70)
    
    print(f"\n⏱️  Total search time: {total_time:.2f}s")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python search_cvs.py <query>")
        print("Example: python search_cvs.py \"Python developer with machine learning\"")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    search_candidates(query)
