from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    index: int
    question: str
    rationale: str = ""


class EvidenceItem(BaseModel):
    evidence_id: str
    title: str
    source_url: str = ""
    source_type: Literal["web", "kb", "manual"] = "web"
    snippet: str = ""
    content: str = ""
    chunk_id: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    query: str
    items: list[EvidenceItem] = Field(default_factory=list)
    total: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    event_type: str
    step_name: str
    status: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    input_summary: str = ""
    output_summary: str = ""
    tool_name: str | None = None
    duration_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewScore(BaseModel):
    factual_support: int = Field(default=0, ge=0, le=5)
    citation_coverage: int = Field(default=0, ge=0, le=5)
    coherence: int = Field(default=0, ge=0, le=5)
    completeness: int = Field(default=0, ge=0, le=5)

    @property
    def total(self) -> int:
        return (
            self.factual_support
            + self.citation_coverage
            + self.coherence
            + self.completeness
        )


class ReviewFeedback(BaseModel):
    missing_subquestions: list[str] = Field(default_factory=list)
    weak_sections: list[str] = Field(default_factory=list)
    low_quality_sources: list[str] = Field(default_factory=list)
    subquestion_source_map: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict
    )


class DraftReview(BaseModel):
    accepted: bool = False
    score: ReviewScore = Field(default_factory=ReviewScore)
    suggestions: list[str] = Field(default_factory=list)
    feedback: ReviewFeedback = Field(default_factory=ReviewFeedback)


class FinalReport(BaseModel):
    topic: str
    outline: list[str] = Field(default_factory=list)
    content: str = ""
    references: list[EvidenceItem] = Field(default_factory=list)
    review: DraftReview | None = None
    writer_trace: list[dict[str, Any]] = Field(default_factory=list)


class ResearchSection(BaseModel):
    sub_question: str
    query_history: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    summary: str = ""
    draft_section: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowResult(BaseModel):
    topic: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    sections: list[ResearchSection] = Field(default_factory=list)
    final_report: FinalReport
    trace: list[TraceEvent] = Field(default_factory=list)
