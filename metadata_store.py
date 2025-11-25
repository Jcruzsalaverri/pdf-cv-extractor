"""
Metadata store for managing structured CV data across multiple candidates.

This module maintains a searchable index of all candidates with their
extracted structured data (skills, experience, education).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from cv_extractor import CVData


class MetadataStore:
    """Manages candidate metadata storage and retrieval."""
    
    def __init__(self, store_path: str = "./cv_metadata.json"):
        """
        Initialize metadata store.
        
        Args:
            store_path (str): Path to metadata JSON file
        """
        self.store_path = Path(store_path)
        self.data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load metadata from file."""
        if self.store_path.exists():
            with open(self.store_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"candidates": {}, "last_updated": None}
    
    def _save(self):
        """Save metadata to file."""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.store_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
    
    def _find_duplicate(self, cv_data: CVData) -> Optional[str]:
        """
        Find duplicate candidate by name or email.
        
        Args:
            cv_data (CVData): Candidate data to check
        
        Returns:
            str or None: Candidate ID of duplicate if found
        """
        name_lower = cv_data.candidate_name.lower().strip()
        email_lower = cv_data.email.lower().strip() if cv_data.email else None
        
        for cid, candidate in self.data["candidates"].items():
            # Check by name (exact match, case-insensitive)
            if candidate.get("candidate_name", "").lower().strip() == name_lower:
                return cid
            
            # Check by email (if both have email)
            if email_lower and candidate.get("email"):
                if candidate.get("email", "").lower().strip() == email_lower:
                    return cid
        
        return None
    
    def add_candidate(self, cv_data: CVData, candidate_id: Optional[str] = None) -> str:
        """
        Add or update candidate in metadata store.
        Automatically detects and removes duplicates (keeps latest).
        
        Args:
            cv_data (CVData): Extracted CV data
            candidate_id (str, optional): Unique ID (auto-generated if not provided)
        
        Returns:
            str: Candidate ID
        """
        # Check for duplicates
        duplicate_id = self._find_duplicate(cv_data)
        
        if duplicate_id:
            print(f"⚠️  Duplicate detected: {cv_data.candidate_name}")
            print(f"   Removing old entry: {duplicate_id}")
            del self.data["candidates"][duplicate_id]
        
        if candidate_id is None:
            # Generate ID from name and timestamp
            name_slug = cv_data.candidate_name.lower().replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            candidate_id = f"{name_slug}_{timestamp}"
        
        # Store candidate data
        self.data["candidates"][candidate_id] = cv_data.to_dict()
        self._save()
        
        if duplicate_id:
            print(f"   ✓ Updated with new entry: {candidate_id}")
        
        return candidate_id
    
    def get_candidate(self, candidate_id: str) -> Optional[CVData]:
        """
        Get candidate data by ID.
        
        Args:
            candidate_id (str): Candidate ID
        
        Returns:
            CVData or None: Candidate data if found
        """
        if candidate_id in self.data["candidates"]:
            return CVData(**self.data["candidates"][candidate_id])
        return None
    
    def search_by_skill(self, skill: str) -> List[Dict[str, Any]]:
        """
        Find candidates with a specific skill.
        
        Args:
            skill (str): Skill to search for (case-insensitive)
        
        Returns:
            List[Dict]: Matching candidates with their IDs
        """
        skill_lower = skill.lower()
        matches = []
        
        for cid, candidate in self.data["candidates"].items():
            all_skills = (
                candidate.get("technical_skills", []) +
                candidate.get("programming_languages", []) +
                candidate.get("frameworks", []) +
                candidate.get("tools", [])
            )
            
            if any(skill_lower in s.lower() for s in all_skills):
                matches.append({
                    "candidate_id": cid,
                    "name": candidate.get("candidate_name"),
                    "current_role": candidate.get("current_role"),
                    "matching_skills": [s for s in all_skills if skill_lower in s.lower()]
                })
        
        return matches
    
    def search_by_experience(self, min_years: float, max_years: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Find candidates by years of experience.
        
        Args:
            min_years (float): Minimum years
            max_years (float, optional): Maximum years
        
        Returns:
            List[Dict]: Matching candidates
        """
        matches = []
        
        for cid, candidate in self.data["candidates"].items():
            years = candidate.get("total_years_experience", 0)
            
            if years >= min_years:
                if max_years is None or years <= max_years:
                    matches.append({
                        "candidate_id": cid,
                        "name": candidate.get("candidate_name"),
                        "years_experience": years,
                        "current_role": candidate.get("current_role")
                    })
        
        # Sort by experience (descending)
        matches.sort(key=lambda x: x["years_experience"], reverse=True)
        return matches
    
    def search_by_company(self, company: str) -> List[Dict[str, Any]]:
        """
        Find candidates who worked at a specific company.
        
        Args:
            company (str): Company name (case-insensitive partial match)
        
        Returns:
            List[Dict]: Matching candidates
        """
        company_lower = company.lower()
        matches = []
        
        for cid, candidate in self.data["candidates"].items():
            companies = candidate.get("companies", [])
            
            if any(company_lower in c.lower() for c in companies):
                matches.append({
                    "candidate_id": cid,
                    "name": candidate.get("candidate_name"),
                    "companies": companies,
                    "current_role": candidate.get("current_role")
                })
        
        return matches
    
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """
        Get all candidates with basic info.
        
        Returns:
            List[Dict]: All candidates
        """
        candidates = []
        
        for cid, candidate in self.data["candidates"].items():
            candidates.append({
                "candidate_id": cid,
                "name": candidate.get("candidate_name"),
                "current_role": candidate.get("current_role"),
                "years_experience": candidate.get("total_years_experience", 0),
                "num_skills": len(candidate.get("technical_skills", [])),
                "source_file": candidate.get("source_file")
            })
        
        return candidates
    
    def deduplicate(self) -> int:
        """
        Remove duplicate candidates from the store.
        Keeps the most recent entry for each unique candidate.
        
        Returns:
            int: Number of duplicates removed
        """
        seen = {}  # {(name_lower, email_lower): candidate_id}
        to_remove = []
        
        # Sort by candidate_id (which includes timestamp) to keep latest
        sorted_candidates = sorted(
            self.data["candidates"].items(),
            key=lambda x: x[0]  # Sort by ID (timestamp in ID)
        )
        
        for cid, candidate in sorted_candidates:
            name_lower = candidate.get("candidate_name", "").lower().strip()
            email_lower = candidate.get("email", "").lower().strip() if candidate.get("email") else None
            
            # Create unique key
            key = (name_lower, email_lower)
            
            if key in seen:
                # This is a duplicate, mark old one for removal
                old_id = seen[key]
                to_remove.append(old_id)
                print(f"⚠️  Duplicate: {candidate.get('candidate_name')}")
                print(f"   Removing: {old_id}")
                print(f"   Keeping:  {cid}")
            
            # Update to keep latest
            seen[key] = cid
        
        # Remove duplicates
        for cid in to_remove:
            if cid in self.data["candidates"]:
                del self.data["candidates"][cid]
        
        if to_remove:
            self._save()
            print(f"\n✓ Removed {len(to_remove)} duplicate(s)")
        
        return len(to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict: Statistics
        """
        total = len(self.data["candidates"])
        
        if total == 0:
            return {"total_candidates": 0}
        
        # Aggregate skills
        all_skills = set()
        all_languages = set()
        all_companies = set()
        
        for candidate in self.data["candidates"].values():
            all_skills.update(candidate.get("technical_skills", []))
            all_languages.update(candidate.get("programming_languages", []))
            all_companies.update(candidate.get("companies", []))
        
        return {
            "total_candidates": total,
            "unique_skills": len(all_skills),
            "unique_languages": len(all_languages),
            "unique_companies": len(all_companies),
            "last_updated": self.data.get("last_updated")
        }


if __name__ == "__main__":
    import sys
    
    # Demo usage
    store = MetadataStore()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "stats":
            stats = store.get_stats()
            print("\n📊 Metadata Store Statistics:")
            print("=" * 60)
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        elif command == "list":
            candidates = store.get_all_candidates()
            print(f"\n👥 All Candidates ({len(candidates)}):")
            print("=" * 60)
            for c in candidates:
                print(f"\n{c['name']}")
                print(f"  ID: {c['candidate_id']}")
                print(f"  Role: {c['current_role']}")
                print(f"  Experience: {c['years_experience']} years")
                print(f"  Skills: {c['num_skills']}")
        
        elif command == "search-skill" and len(sys.argv) > 2:
            skill = sys.argv[2]
            matches = store.search_by_skill(skill)
            print(f"\n🔍 Candidates with '{skill}' ({len(matches)}):")
            print("=" * 60)
            for m in matches:
                print(f"\n{m['name']}")
                print(f"  Role: {m['current_role']}")
                print(f"  Matching skills: {', '.join(m['matching_skills'])}")
        
        elif command == "deduplicate":
            print("\n🔍 Checking for duplicates...")
            print("=" * 60)
            removed = store.deduplicate()
            if removed == 0:
                print("✓ No duplicates found")
        
        else:
            print("Usage:")
            print("  python metadata_store.py stats")
            print("  python metadata_store.py list")
            print("  python metadata_store.py search-skill Python")
            print("  python metadata_store.py deduplicate")
    else:
        stats = store.get_stats()
        print(f"\n📊 Database has {stats['total_candidates']} candidates")
        print("\nUsage:")
        print("  python metadata_store.py stats")
        print("  python metadata_store.py list")
        print("  python metadata_store.py search-skill <skill>")
