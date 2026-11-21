"""Load and validate ``eval/dataset/questions.jsonl`` (CONTRACT.md section 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DATASET_PATH = Path(__file__).parent / "dataset" / "questions.jsonl"
SMOKE_MIN, SMOKE_MAX, SMOKE_MIN_PER_CLASS = 15, 20, 6


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    smoke: bool
    reference_notes: str


def load_questions(path: Path = DATASET_PATH) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    questions = [Question.model_validate_json(line) for line in lines if line.strip()]
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate question ids in dataset")
    return questions


def validate_smoke_subset(questions: list[Question]) -> None:
    """Contract: 15-20 smoke questions, at least 6 answerable and 6 unanswerable."""
    smoke = [q for q in questions if q.smoke]
    answerable = sum(q.answerable for q in smoke)
    unanswerable = len(smoke) - answerable
    if not SMOKE_MIN <= len(smoke) <= SMOKE_MAX:
        raise ValueError(
            f"smoke subset must have {SMOKE_MIN}-{SMOKE_MAX} questions, has {len(smoke)}"
        )
    if min(answerable, unanswerable) < SMOKE_MIN_PER_CLASS:
        raise ValueError(
            f"smoke subset needs >= {SMOKE_MIN_PER_CLASS} answerable and unanswerable "
            f"(has {answerable}/{unanswerable})"
        )


def select(questions: list[Question], mode: Literal["smoke", "full"]) -> list[Question]:
    return [q for q in questions if q.smoke] if mode == "smoke" else list(questions)
