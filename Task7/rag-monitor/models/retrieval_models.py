from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkMeta:
    chunk_id: str | None = None
    doc_id: int | None = None
    n_tokens: int | None = None
    section_path: str | None = None  # у вас это строка (как в логах)


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    distance: float | None
    text: str
    meta: ChunkMeta


@dataclass(frozen=True)
class RetrievalLogRow:
    """
    Строка из _runs/retrieval_logs/<run_id>_raw_retrieval.jsonl
    """
    run_id: str
    question_id: str
    question: str
    k: int
    results: list[RetrievalResult]
    raw: dict[str, Any] | None = None