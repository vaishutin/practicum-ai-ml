from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from services.rag_api_safe_client import RagApiSafeClient


@dataclass(frozen=True)
class GoldenRetrievalSummary:
    k: int
    questions: int
    raw_log_path: str


class GoldenRetrievalRunner:
    """
    Делает retrieval-only прогон golden вопросов через rag-api-safe bulk endpoint
    и пишет сырые логи в JSONL: 1 строка = 1 вопрос.
    """

    def __init__(self, questions_path: Path, logs_dir: Path, client: RagApiSafeClient) -> None:
        self._questions_path = questions_path
        self._logs_dir = logs_dir
        self._client = client

    def run(self, run_id: str, k: int) -> GoldenRetrievalSummary:
        questions = list(self._load_questions())  # [{"id":..,"question":..}, ...]
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        out_file = self._logs_dir / f"{run_id}_raw_retrieval.jsonl"

        # 1) одним запросом получаем все top-k
        resp = self._client.retrieval_topk_bulk(k=k, questions=questions)
        results = resp.get("results", [])

        # 2) пишем 1 строка = 1 вопрос
        by_id = {r.get("id"): r.get("results", []) for r in results}

        with out_file.open("w", encoding="utf-8") as f:
            for q in questions:
                qid = q["id"]
                question = q["question"]

                payload = {
                    "run_id": run_id,
                    "question_id": qid,
                    "question": question,
                    "k": k,
                    "results": by_id.get(qid, []),  # уже rank/distance/meta/text
                }
                f.write(json.dumps(payload, ensure_ascii=False))
                f.write("\n")

        return GoldenRetrievalSummary(k=k, questions=len(questions), raw_log_path=str(out_file))

    def _load_questions(self):
        if not self._questions_path.exists():
            raise FileNotFoundError(f"Golden questions file not found: {self._questions_path}")

        with self._questions_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = obj.get("id")
                question = obj.get("question")
                if not qid or not question:
                    raise ValueError(f"Golden question must have 'id' and 'question'. Line {line_no}")
                yield {"id": qid, "question": question}