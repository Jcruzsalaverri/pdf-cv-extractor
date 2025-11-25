"""
Structured information extraction from CV/Resume text using LLM.

Extracts:
- Personal info (name, contact)
- Skills (technical, programming languages, frameworks)
- Experience (years, companies, roles)
- Education (degrees, universities, certifications)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import re
from pydantic import BaseModel, Field, field_validator, ConfigDict, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from config import Config


class CVData(BaseModel):
    """Structured CV data schema with validation."""
    
    # Pydantic configuration
    model_config = ConfigDict(
        extra='ignore',  # Ignore extra fields from LLM
        str_strip_whitespace=True,  # Auto-strip whitespace
        validate_assignment=True  # Validate on attribute assignment
    )
    
    # Identity
    candidate_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    
    # Experience
    total_years_experience: Optional[float] = Field(default=0.0, description="Total years of professional experience")
    current_role: Optional[str] = None
    companies: List[str] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    
    # Skills
    technical_skills: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    
    # Education
    degrees: List[Dict[str, Any]] = Field(default_factory=list, description="List of degrees with degree, field, university, year")
    certifications: List[str] = Field(default_factory=list)
    
    # Metadata
    source_file: str = ""
    processed_date: str = ""
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Basic email validation and cleanup."""
        if v is None:
            return v
        # Remove spaces
        v = v.replace(' ', '')
        # Basic email format check
        if v and '@' not in v:
            return None  # Invalid email, set to None
        return v.lower() if v else None
    
    @field_validator('linkedin')
    @classmethod
    def validate_linkedin(cls, v: Optional[str]) -> Optional[str]:
        """Clean LinkedIn URL."""
        if v is None:
            return v
        # Remove spaces
        v = v.replace(' ', '')
        return v if v else None
    
    @field_validator('total_years_experience')
    @classmethod
    def validate_experience(cls, v: Optional[float]) -> Optional[float]:
        """Ensure experience is reasonable."""
        if v is None:
            return None
        if v < 0:
            return 0.0
        if v > 70:  # Sanity check
            return 70.0
        return round(v, 1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    def to_json(self) -> str:
        """Convert to JSON string with proper Unicode handling."""
        return self.model_dump_json(indent=2, exclude_none=False)


def initialize_extraction_llm(provider: str = None, temperature: float = 0.1):
    """
    Initialize LLM for extraction (low temperature for consistency).
    
    Args:
        provider (str, optional): "gemini" or "claude"
        temperature (float): Low temperature for factual extraction
    
    Returns:
        LLM instance
    """
    provider = provider or Config.LLM_PROVIDER
    
    if provider == "gemini":
        if not Config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set")
        
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
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


def extract_cv_data(cv_text: str, source_file: str = "", provider: str = None) -> CVData:
    """
    Extract structured data from CV text using LLM.
    
    Args:
        cv_text (str): Cleaned CV text
        source_file (str): Source filename
        provider (str, optional): LLM provider
    
    Returns:
        CVData: Extracted structured data
    """
    llm = initialize_extraction_llm(provider)
    
    system_prompt = """You are a CV/Resume data extraction assistant.

Your task is to extract structured information from CV text and output it as valid JSON.

Extract the following fields:
- candidate_name (string): Full name
- email (string or null): Email address
- phone (string or null): Phone number
- linkedin (string or null): LinkedIn URL
- total_years_experience (number): Total years of professional experience
- current_role (string or null): Current job title (if multiple, keep as combined)
- companies (array of strings): List of all companies worked at
- roles (array of strings): List of all job titles/roles (if a role contains | / or similar, split into separate entries)
- technical_skills (array of strings): All technical skills mentioned
- programming_languages (array of strings): Programming languages (Python, Java, etc.)
- frameworks (array of strings): Frameworks and libraries (Django, React, etc.)
- tools (array of strings): Tools and platforms (AWS, Docker, Git, etc.)
- degrees (array of objects): Each with {degree, field, university, year}
- certifications (array of strings): Certifications and licenses

Rules:
1. Extract ALL information present in the CV
2. For years of experience, calculate from date ranges if provided
3. Categorize skills appropriately (languages vs frameworks vs tools)
4. For roles: if you see "Product Support | Product Owner" or "Developer / Designer", split into ["Product Support", "Product Owner"] or ["Developer", "Designer"]
5. Output ONLY valid JSON, no explanations
6. Use null for missing fields, empty arrays [] for missing lists"""

    user_prompt = f"""Extract structured data from this CV:

{cv_text}

Output (JSON only):"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        print("🔍 Extracting structured data from CV...")
        response = llm.invoke(messages)
        json_text = response.content.strip()
        
        # Clean JSON (remove markdown code blocks if present)
        if json_text.startswith("```"):
            json_text = re.sub(r'```json\n?', '', json_text)
            json_text = re.sub(r'```\n?', '', json_text)
        
        # Parse JSON
        data_dict = json.loads(json_text)
        
        # Add metadata
        data_dict['source_file'] = source_file
        data_dict['processed_date'] = datetime.now().isoformat()
        
        # Create CVData object (Pydantic handles validation and cleanup)
        cv_data = CVData(**data_dict)
        
        print(f"✓ Extracted data for: {cv_data.candidate_name}")
        print(f"  Skills: {len(cv_data.technical_skills)} technical skills")
        print(f"  Languages: {len(cv_data.programming_languages)} programming languages")
        print(f"  Experience: {cv_data.total_years_experience} years")
        print(f"  Companies: {len(cv_data.companies)}")
        print(f"  Education: {len(cv_data.degrees)} degrees")
        
        return cv_data
        
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse JSON: {e}")
        print(f"   Response: {json_text[:200]}...")
        # Return minimal CVData
        return CVData(
            candidate_name="Unknown",
            source_file=source_file,
            processed_date=datetime.now().isoformat()
        )
    except ValidationError as e:
        print(f"⚠️  Pydantic validation failed: {e}")
        print(f"   Errors: {e.errors()}")
        print(f"   JSON data: {json.dumps(data_dict, indent=2)[:500]}...")
        # Return minimal CVData
        return CVData(
            candidate_name="Unknown",
            source_file=source_file,
            processed_date=datetime.now().isoformat()
        )
    except Exception as e:
        print(f"⚠️  Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return CVData(
            candidate_name="Unknown",
            source_file=source_file,
            processed_date=datetime.now().isoformat()
        )



if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Extract structured data from CV text")
    parser.add_argument('input_file', help='Path to CV text file')
    parser.add_argument('--provider', choices=['gemini', 'claude'], default=None,
                       help='LLM provider (default: from config)')
    
    args = parser.parse_args()
    
    if not Path(args.input_file).exists():
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    
    # Read CV text
    with open(args.input_file, 'r', encoding='utf-8') as f:
        cv_text = f.read()
    
    # Extract data
    cv_data = extract_cv_data(cv_text, source_file=args.input_file, provider=args.provider)
    
    # Save JSON
    output_file = args.input_file.replace('.txt', '_extracted.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cv_data.to_json())
    
    print(f"\n✓ Extracted data saved to: {output_file}")
    
    # Display summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY:")
    print(f"{'='*70}\n")
    print(f"Name: {cv_data.candidate_name}")
    print(f"Email: {cv_data.email}")
    print(f"Current Role: {cv_data.current_role}")
    print(f"Experience: {cv_data.total_years_experience} years")
    print(f"\nSkills ({len(cv_data.technical_skills)}):")
    for skill in cv_data.technical_skills[:10]:
        print(f"  - {skill}")
    if len(cv_data.technical_skills) > 10:
        print(f"  ... and {len(cv_data.technical_skills) - 10} more")
    
    print(f"\nProgramming Languages ({len(cv_data.programming_languages)}):")
    for lang in cv_data.programming_languages:
        print(f"  - {lang}")
    
    print(f"\nCompanies ({len(cv_data.companies)}):")
    for company in cv_data.companies:
        print(f"  - {company}")
    
    print(f"\nEducation ({len(cv_data.degrees)}):")
    for degree in cv_data.degrees:
        print(f"  - {degree.get('degree', 'N/A')} in {degree.get('field', 'N/A')}")
        print(f"    {degree.get('university', 'N/A')} ({degree.get('year', 'N/A')})")
