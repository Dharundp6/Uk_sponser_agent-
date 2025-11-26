"""
Data models for UK Visa Sponsorship Assistant
Contains UserProfile and AgentMemory classes
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class UserProfile:
    """User profile extracted from resume"""
    name: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experience_years: Optional[float] = None
    education: List[str] = field(default_factory=list)
    job_titles: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    resume_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert UserProfile to dictionary for serialization"""
        return {
            "name": self.name,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "education": self.education,
            "job_titles": self.job_titles,
            "industries": self.industries
        }


@dataclass
class AgentMemory:
    """Agent's working memory for context management"""
    user_profile: Optional[UserProfile] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_history: List[Dict[str, Any]] = field(default_factory=list)
    last_search_results: Optional[List[Dict]] = None
    context_notes: List[str] = field(default_factory=list)
    iteration_count: int = 0
