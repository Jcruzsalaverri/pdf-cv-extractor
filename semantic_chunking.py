"""
Semantic chunking for CV data based on structured fields.

Creates meaningful chunks from CVData Pydantic model instead of arbitrary token splits.
"""

from typing import List
from chunking import TextChunk
from cv_extractor import CVData


def create_semantic_chunks_from_cvdata(cv_data: CVData, source_file: str = None) -> List[TextChunk]:
    """
    Create semantic chunks based on CVData structure.
    
    Each chunk represents a logical section of the CV (contact, experience, skills, education).
    This provides better semantic coherence than token-based chunking.
    
    Args:
        cv_data (CVData): Structured CV data from Pydantic model
        source_file (str, optional): Source PDF filename
    
    Returns:
        List[TextChunk]: List of semantically meaningful chunks
    """
    chunks = []
    chunk_id = 0
    
    base_metadata = {
        "candidate_name": cv_data.candidate_name,
        "source_file": source_file or cv_data.source_file,
        "chunking_method": "semantic"
    }
    
    # Chunk 1: Contact & Summary Information
    contact_parts = [f"Candidate: {cv_data.candidate_name}"]
    
    if cv_data.email:
        contact_parts.append(f"Email: {cv_data.email}")
    if cv_data.phone:
        contact_parts.append(f"Phone: {cv_data.phone}")
    if cv_data.linkedin:
        contact_parts.append(f"LinkedIn: {cv_data.linkedin}")
    if cv_data.current_role:
        contact_parts.append(f"Current Role: {cv_data.current_role}")
    if cv_data.total_years_experience:
        contact_parts.append(f"Total Experience: {cv_data.total_years_experience} years")
    
    if contact_parts:
        chunks.append(TextChunk(
            text="\n".join(contact_parts),
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "contact_summary"}
        ))
        chunk_id += 1
    
    # Chunk 2: Work Experience
    if cv_data.companies or cv_data.roles:
        exp_parts = ["Work Experience:"]
        
        if cv_data.total_years_experience:
            exp_parts.append(f"Total: {cv_data.total_years_experience} years")
        
        if cv_data.current_role:
            exp_parts.append(f"Current: {cv_data.current_role}")
        
        if cv_data.companies:
            exp_parts.append(f"Companies: {', '.join(cv_data.companies)}")
        
        if cv_data.roles:
            exp_parts.append(f"Roles: {', '.join(cv_data.roles)}")
        
        chunks.append(TextChunk(
            text="\n".join(exp_parts),
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "experience"}
        ))
        chunk_id += 1
    
    # Chunk 2: Contextual Technical Profile (Experience-Anchored)
    # Ties skills to role context and experience level for better semantic matching
    tech_profile_parts = []
    
    if cv_data.programming_languages or cv_data.frameworks or cv_data.tools or cv_data.technical_skills:
        # Build role context
        role_context = []
        if cv_data.current_role:
            role_context.append(f"Current Role: {cv_data.current_role}")
        if cv_data.total_years_experience:
            role_context.append(f"Experience: {cv_data.total_years_experience} years")
        
        if role_context:
            tech_profile_parts.append(" | ".join(role_context))
        
        # Skill clustering: Group related technologies semantically
        ml_stack = []
        data_stack = []
        cloud_stack = []
        web_stack = []
        other_skills = []
        
        all_skills = (cv_data.programming_languages + cv_data.frameworks + 
                     cv_data.tools + cv_data.technical_skills)
        
        for skill in all_skills:
            skill_upper = skill.upper()
            if any(kw in skill_upper for kw in ['TENSORFLOW', 'PYTORCH', 'SCIKIT', 'KERAS', 'ML', 'MACHINE LEARNING', 'AI']):
                ml_stack.append(skill)
            elif any(kw in skill_upper for kw in ['SQL', 'SPARK', 'AIRFLOW', 'DBT', 'KAFKA', 'REDSHIFT', 'BIGQUERY', 'DATA']):
                data_stack.append(skill)
            elif any(kw in skill_upper for kw in ['AWS', 'AZURE', 'GCP', 'CLOUD', 'LAMBDA', 'S3', 'EC2', 'KUBERNETES', 'DOCKER']):
                cloud_stack.append(skill)
            elif any(kw in skill_upper for kw in ['REACT', 'ANGULAR', 'VUE', 'DJANGO', 'FLASK', 'FASTAPI', 'REST', 'API']):
                web_stack.append(skill)
            else:
                other_skills.append(skill)
        
        # Build contextual skill descriptions
        if ml_stack:
            tech_profile_parts.append(f"ML/AI Stack: {', '.join(ml_stack)}")
        if data_stack:
            tech_profile_parts.append(f"Data Engineering: {', '.join(data_stack)}")
        if cloud_stack:
            tech_profile_parts.append(f"Cloud & Infrastructure: {', '.join(cloud_stack)}")
        if web_stack:
            tech_profile_parts.append(f"Web Development: {', '.join(web_stack)}")
        if cv_data.programming_languages:
            tech_profile_parts.append(f"Programming Languages: {', '.join(cv_data.programming_languages)}")
        if other_skills:
            tech_profile_parts.append(f"Other Technologies: {', '.join(other_skills)}")
        
        # Add education context with technical keywords
        if cv_data.degrees:
            edu_highlights = []
            for degree in cv_data.degrees:
                field = degree.get('field', '')
                degree_name = degree.get('degree', '')
                if field and any(keyword in field.upper() for keyword in ['MACHINE LEARNING', 'DATA', 'COMPUTER', 'ENGINEERING', 'AI', 'SOFTWARE']):
                    edu_highlights.append(f"{degree_name} in {field}")
            if edu_highlights:
                tech_profile_parts.append(f"Education: {' | '.join(edu_highlights)}")
        
        # Add company context for credibility signals
        if cv_data.companies:
            tech_profile_parts.append(f"Companies: {', '.join(cv_data.companies[:3])}")  # Top 3
        
        chunks.append(TextChunk(
            text="\n".join(tech_profile_parts),
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "technical_profile"}
        ))
        chunk_id += 1
    
    # Chunk 3: Programming Languages (detailed, for specific language searches)
    if cv_data.programming_languages:
        skills_text = f"Programming Languages: {', '.join(cv_data.programming_languages)}"
        
        chunks.append(TextChunk(
            text=skills_text,
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "programming_languages"}
        ))
        chunk_id += 1
    
    # Chunk 4: Frameworks & Libraries
    if cv_data.frameworks:
        frameworks_text = f"Frameworks & Libraries: {', '.join(cv_data.frameworks)}"
        
        chunks.append(TextChunk(
            text=frameworks_text,
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "frameworks"}
        ))
        chunk_id += 1
    
    # Chunk 5: Tools & Technologies
    if cv_data.tools:
        tools_text = f"Tools & Technologies: {', '.join(cv_data.tools)}"
        
        chunks.append(TextChunk(
            text=tools_text,
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "tools"}
        ))
        chunk_id += 1
    
    # Chunk 6: General Technical Skills
    if cv_data.technical_skills:
        tech_skills_text = f"Technical Skills: {', '.join(cv_data.technical_skills)}"
        
        chunks.append(TextChunk(
            text=tech_skills_text,
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "technical_skills"}
        ))
        chunk_id += 1
    
    # Chunk 7: Education
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
                edu_parts.append("- " + " ".join(degree_str_parts))
        
        chunks.append(TextChunk(
            text="\n".join(edu_parts),
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "education"}
        ))
        chunk_id += 1
    
    # Chunk 8: Certifications
    if cv_data.certifications:
        cert_text = f"Certifications: {', '.join(cv_data.certifications)}"
        
        chunks.append(TextChunk(
            text=cert_text,
            chunk_id=chunk_id,
            metadata={**base_metadata, "section": "certifications"}
        ))
        chunk_id += 1
    
    return chunks


def create_full_cv_chunk(cv_data: CVData, cleaned_text: str, source_file: str = None) -> TextChunk:
    """
    Create a single chunk containing the full cleaned CV text.
    
    Useful for general semantic search where full context is needed.
    
    Args:
        cv_data (CVData): Structured CV data (for metadata)
        cleaned_text (str): Full cleaned CV text
        source_file (str, optional): Source PDF filename
    
    Returns:
        TextChunk: Single chunk with full CV text
    """
    metadata = {
        "candidate_name": cv_data.candidate_name,
        "source_file": source_file or cv_data.source_file,
        "chunking_method": "full_document",
        "section": "full_cv"
    }
    
    return TextChunk(
        text=cleaned_text,
        chunk_id=0,
        metadata=metadata
    )
