from typing import Optional, List
from pydantic import BaseModel, Field


class HRState(BaseModel):
    """
    Central state object passed between all agents in the system.
    This is the single source of truth for the workflow.
    """

    # Original user input
    user_query: str

    # Intent decided by classifier (e.g., policy, benefits, procedures)
    intent: Optional[str] = None

    # Retrieved evidence snippets (filled by RAG later)
    # evidence: List[str] = []

    # Final answer produced by an agent
    answer: Optional[str] = None

    # Debug / trace metadata
    # trace: List[str] = []
    
    # sources: list[str] = Field(default_factory=list)

    sources: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
