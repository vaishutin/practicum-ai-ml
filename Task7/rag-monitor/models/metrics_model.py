from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RetrievalMetricsRow:
    """
    Пишем в metrics/retrieval_metrics.jsonl
    """
    ts: datetime
    run_id: str
    question_id: str
    k: int

    # базовые retrieval метрики
    hit: float           # 0/1
    recall: float        # 0..1
    mrr: float           # 0..1


@dataclass(frozen=True)
class RagasMetricsRow:
    """
    Пишем в metrics/ragas_metrics.jsonl
    """
    ts: datetime
    run_id: str
    question_id: str
    model: str  # модель-судья/генератор

    # основные метрики
    faithfulness: float
    answer_relevancy: float

    # режим cot / concise / etc
    mode: Optional[str] = None