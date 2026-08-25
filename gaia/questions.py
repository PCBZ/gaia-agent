"""Fetch GAIA questions from the course API, cache them locally, model with Pydantic.

Usage:
    from gaia.questions import load_questions
    questions = load_questions()              # local cache if present, else fetch + save
    questions = load_questions(refresh=True)  # force re-fetch from the API
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from pydantic import BaseModel, ConfigDict, Field

SCORING_API = "https://agents-course-unit4-scoring.hf.space"
# questions.json lives at the repo root (one level above this package).
CACHE_FILE = Path(__file__).parent.parent / "questions.json"


class Question(BaseModel):
    """One GAIA question as served by the course scoring API."""

    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    question: str
    level: str = Field(alias="Level")
    file_name: str = ""

    @property
    def has_file(self) -> bool:
        return bool(self.file_name)


def fetch_questions(api: str = SCORING_API, timeout: float = 60.0) -> list[Question]:
    """GET /questions from the API and parse into Question models."""
    resp = requests.get(f"{api}/questions", timeout=timeout)
    resp.raise_for_status()
    return [Question.model_validate(item) for item in resp.json()]


def save_questions(questions: list[Question], cache: Path = CACHE_FILE) -> None:
    """Write questions to the local JSON cache (keeping the API's field names)."""
    data = [q.model_dump(by_alias=True) for q in questions]
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(cache: Path = CACHE_FILE, refresh: bool = False) -> list[Question]:
    """Return the questions.

    If the local cache exists (and refresh is False), parse it with Pydantic.
    Otherwise fetch from the API and save the cache.
    """
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return [Question.model_validate(item) for item in data]

    questions = fetch_questions()
    save_questions(questions, cache)
    return questions
