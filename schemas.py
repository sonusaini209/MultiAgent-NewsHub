"""Data models."""
from typing import List, Literal

from pydantic import BaseModel, Field

TaskName = Literal["blog", "summary", "categories", "trends"]


class NewsArticle(BaseModel):
    title: str = ""
    description: str = ""
    source: str = "Unknown"
    url: str = ""
    published_at: str = ""
    content: str = ""


# ── Structured output of the LLM agents ────────────────────────────────
class SupervisorDecision(BaseModel):
    tasks: List[TaskName] = Field(description="Analysis tasks worth running for this topic")
    reasoning: str = Field(description="One or two sentences explaining the choice")


class QueryRewrite(BaseModel):
    query: str = Field(description="A broader news search query on the same topic")


class TaskVerdict(BaseModel):
    task: TaskName
    passed: bool
    issues: str = Field(default="", description="What is wrong, if anything")


class ReviewResult(BaseModel):
    verdicts: List[TaskVerdict]
