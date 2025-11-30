"""
LLM-based text cleaning and normalization for CV/Resume processing.

This module cleans raw PDF text to improve embedding quality and retrieval accuracy.
"""

from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config
import re


def initialize_cleaning_llm(provider: str = None, temperature: float = 0.1, api_key: str = None):
    """
    Initialize LLM for text cleaning (low temperature for consistency).
    
    Args:
        provider (str, optional): "gemini" or "claude"
        temperature (float): Low temperature for consistent cleaning
        api_key (str, optional): Specific API key to use (for rotation)
    
    Returns:
        LLM instance
    """
    provider = provider or Config.LLM_PROVIDER
    
    if provider == "gemini":
        # Use provided key or get from config
        gemini_key = api_key or Config.GEMINI_API_KEY
        
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not set")
        
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=gemini_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
        
    elif provider == "claude":
        if not Config.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        llm = ChatAnthropic(
            model=Config.CLAUDE_MODEL,
            anthropic_api_key=Config.ANTHROPIC_API_KEY,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    
    return llm


def clean_cv_text(raw_text: str, provider: str = None, use_llm: bool = True, api_key: str = None) -> str:
    """
    Clean and normalize CV text to improve embedding quality.
    
    Args:
        raw_text (str): Raw text extracted from PDF
        provider (str, optional): LLM provider
        use_llm (bool): Whether to use LLM cleaning (True) or rule-based (False)
        api_key (str, optional): Specific API key to use (for rotation)
    
    Returns:
        str: Cleaned and normalized text
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    # Quick rule-based pre-cleaning
    text = _rule_based_cleaning(raw_text)
    
    # LLM-based deep cleaning (optional but recommended)
    if use_llm:
        text = _llm_based_cleaning(text, provider, api_key=api_key)
    
    return text


def _rule_based_cleaning(text: str) -> str:
    """
    Apply rule-based cleaning for common PDF artifacts.
    
    Args:
        text (str): Raw text
    
    Returns:
        str: Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spaced-out words (e.g., "W O R K" -> "WORK")
    # Match single letters separated by spaces
    text = re.sub(r'\b([A-Z])\s+(?=[A-Z]\s)', r'\1', text)
    
    # Remove special Unicode characters that don't render well
    text = text.replace('\u2022', '•')  # Bullet points
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '-')  # Em dash
    text = text.replace('\u2019', "'")  # Right single quote
    text = text.replace('\u201c', '"')  # Left double quote
    text = text.replace('\u201d', '"')  # Right double quote
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def _llm_based_cleaning(text: str, provider: str = None, api_key: str = None) -> str:
    """
    Use LLM to deeply clean and normalize CV text.
    
    Args:
        text (str): Pre-cleaned text
        provider (str, optional): LLM provider
        api_key (str, optional): Specific API key to use
    
    Returns:
        str: LLM-cleaned text
    """
    llm = initialize_cleaning_llm(provider, api_key=api_key)
    
    system_prompt = """You are a text normalization assistant for CV/Resume processing.

Your task is to clean and normalize CV text while preserving ALL information.

Rules:
1. Remove extra spaces between letters (e.g., "W O R K" -> "WORK")
2. Fix formatting artifacts and encoding issues
3. Standardize section headers to proper case (e.g., "WORK EXPERIENCE", "EDUCATION", "SKILLS")
4. Preserve all dates, names, companies, and details EXACTLY as written
5. Maintain proper paragraph structure
6. Keep bullet points and lists
7. Output ONLY the cleaned text, no explanations or comments

Important: Do NOT summarize, rephrase, or omit any information. Only fix formatting."""

    user_prompt = f"""Clean this CV text:

{text}

Output (cleaned text only):"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = llm.invoke(messages)
        cleaned_text = response.content.strip()
        
        # Validate that cleaning didn't lose too much content
        if len(cleaned_text) < len(text) * 0.5:
            print("⚠️  Warning: LLM cleaning removed >50% of text, using rule-based only")
            return text
        
        return cleaned_text
        
    except Exception as e:
        print(f"⚠️  LLM cleaning failed: {e}")
        print("   Falling back to rule-based cleaning")
        return text





def clean_and_structure_cv(raw_text: str, provider: str = None, api_key: str = None) -> Dict[str, any]:
    """
    Complete cleaning pipeline for CV text.
    
    Args:
        raw_text (str): Raw PDF text
        provider (str, optional): LLM provider
        api_key (str, optional): Specific API key to use (for rotation)
    
    Returns:
        Dict containing cleaned_text and metadata
    """
    print("🧹 Cleaning CV text...")
    
    # Clean text
    cleaned_text = clean_cv_text(raw_text, provider=provider, use_llm=True, api_key=api_key)
    
    print(f"   Original length: {len(raw_text)} chars")
    print(f"   Cleaned length: {len(cleaned_text)} chars")
    
    return {
        'cleaned_text': cleaned_text,
        'original_length': len(raw_text),
        'cleaned_length': len(cleaned_text)
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python text_cleaner.py <pdf_file_or_text_file>")
        print("\nExample:")
        print("  python text_cleaner.py CV_Javier_Cruz_Nov25.pdf")
        print("  python text_cleaner.py CV_Javier_Cruz_Nov25.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Read input
    if input_file.endswith('.pdf'):
        from extract_text import extract_text_from_pdf
        raw_text = extract_text_from_pdf(input_file)
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    
    # Clean and structure
    result = clean_and_structure_cv(raw_text)
    
    # Save cleaned text
    output_file = input_file.replace('.pdf', '_cleaned.txt').replace('.txt', '_cleaned.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result['cleaned_text'])
    
    print(f"\n✓ Cleaned text saved to: {output_file}")
    
    # Show preview of cleaned text
    print(f"\n{'='*70}")
    print("CLEANED TEXT PREVIEW:")
    print(f"{'='*70}\n")
    print(result['cleaned_text'][:500])
    if len(result['cleaned_text']) > 500:
        print("\n[... truncated ...]")
