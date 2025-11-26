"""
CV data formatting and chunk representation.

Formats CVData into a single comprehensive text chunk for vector search.
"""

from typing import Dict, Any, List
from cv_extractor import CVData


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


def format_cv_for_search(cv_data: CVData, source_file: str = "") -> List[TextChunk]:
    """
    Format CV data into a single comprehensive text chunk.
    
    Combines all CV information into one searchable text block, providing
    full context to the embedding model for better semantic matching.
    
    Args:
        cv_data (CVData): Structured CV data from Pydantic model
        source_file (str, optional): Source PDF filename
    
    Returns:
        List[TextChunk]: List containing single comprehensive chunk
    """
    sections = []
    
    # Identity & Summary
    sections.append(f"Candidate: {cv_data.candidate_name}")
    
    if cv_data.current_role:
        sections.append(f"Current Role: {cv_data.current_role}")
    
    if cv_data.total_years_experience:
        sections.append(f"Total Experience: {cv_data.total_years_experience} years")
    
    # Contact Information
    contact_info = []
    if cv_data.email:
        contact_info.append(f"Email: {cv_data.email}")
    if cv_data.phone:
        contact_info.append(f"Phone: {cv_data.phone}")
    if cv_data.linkedin:
        contact_info.append(f"LinkedIn: {cv_data.linkedin}")
    
    if contact_info:
        sections.append("\n".join(contact_info))
    
    # Technical Skills
    if cv_data.programming_languages:
        sections.append(f"Programming Languages: {', '.join(cv_data.programming_languages)}")
    
    if cv_data.frameworks:
        sections.append(f"Frameworks & Libraries: {', '.join(cv_data.frameworks)}")
    
    if cv_data.tools:
        sections.append(f"Tools & Technologies: {', '.join(cv_data.tools)}")
    
    if cv_data.technical_skills:
        sections.append(f"Technical Skills: {', '.join(cv_data.technical_skills)}")
    
    # Work Experience
    if cv_data.companies:
        sections.append(f"Companies: {', '.join(cv_data.companies)}")
    
    if cv_data.roles:
        sections.append(f"Roles: {', '.join(cv_data.roles)}")
    
    # Education
    if cv_data.degrees:
        edu_parts = ["Education:"]
        for degree in cv_data.degrees:
            degree_str_parts = []
            if degree.get('degree'):
                degree_str_parts.append(degree['degree'])
            if degree.get('field'):
                degree_str_parts.append(f"in {degree['field']}")
            if degree.get('university'):
                degree_str_parts.append(f"from {degree['university']}")
            if degree.get('year'):
                degree_str_parts.append(f"({degree['year']})")
            
            if degree_str_parts:
                edu_parts.append("  " + " ".join(degree_str_parts))
        
        sections.append("\n".join(edu_parts))
    
    # Certifications
    if cv_data.certifications:
        sections.append(f"Certifications: {', '.join(cv_data.certifications)}")
    
    # Combine all sections
    full_text = "\n\n".join(sections)
    
    # Create metadata
    metadata = {
        "source_file": source_file or cv_data.source_file,
        "candidate_name": cv_data.candidate_name,
        "chunk_type": "full_profile",
        "total_years_experience": cv_data.total_years_experience or 0.0,
        "current_role": cv_data.current_role or "",
        "technical_skills": ",".join(cv_data.technical_skills) if cv_data.technical_skills else "",
        "programming_languages": ",".join(cv_data.programming_languages) if cv_data.programming_languages else "",
        "companies": ",".join(cv_data.companies) if cv_data.companies else ""
    }
    
    # Create chunk
    chunk = TextChunk(
        chunk_id=f"{cv_data.candidate_name.replace(' ', '_')}_full_profile",
        text=full_text,
        metadata=metadata
    )
    
    return [chunk]