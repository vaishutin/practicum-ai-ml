from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class GoldenRetrievalRef:
    k: int
    questions: int
    raw_log_path: str


@dataclass(frozen=True)
class RunMeta:
    """
    Строка из rag-kb-updater/_runs/runs.jsonl.

    Важно: golden_retrieval может быть None.
    Остальные поля (кроме run_id) мы держим "как есть" в raw, чтобы не ломаться от эволюции формата.
    """
    run_id: str
    created_at: Optional[datetime] = None
    golden_retrieval: Optional[GoldenRetrievalRef] = None
    raw: dict[str, Any] | None = None


@dataclass(frozen=True)
class RunProcessState:
    """
    То, что пишет rag-monitor (какие run_id уже обработаны).
    """
    run_id: str
    processed_at: datetime
    # для дебага/статистики:
    retrieval_metrics_written: bool = False
    ragas_metrics_written: bool = False
    notes: str | None = None