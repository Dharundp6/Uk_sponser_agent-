"""
UK Visa Sponsorship Assistant - Modular Package
A fully agentic AI assistant for UK Skilled Worker visa sponsorship and job search
"""

__version__ = "2.0.0"
__author__ = "UK RAG Team"

# Main exports
from agent import AgenticVisaAssistant
from vector_db import EnhancedVectorDatabase
from models import UserProfile, AgentMemory
from tools import ToolImplementations

__all__ = [
    "AgenticVisaAssistant",
    "EnhancedVectorDatabase",
    "UserProfile",
    "AgentMemory",
    "ToolImplementations"
]
