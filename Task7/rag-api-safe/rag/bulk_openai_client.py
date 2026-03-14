# rag_api/bulk_llm_client.py
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

logger = logging.getLogger("bulk-llm-client")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
BULK_WORKERS = int(os.getenv("BULK_LLM_WORKERS", "10"))


class BulkOpenAIClient:
    def __init__(self, openai_client):
        self._client = openai_client

    def _call_one(self, question_id: str, instruction: str, context_with_question: str) -> Dict[str, Any]:
        try:
            resp = self._client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": context_with_question},
                ],
            )
            return {"question_id": question_id, "answer": resp.choices[0].message.content, "error": None}
        except Exception as e:
            logger.exception("LLM error for question_id=%s", question_id)
            return {"question_id": question_id, "answer": None, "error": str(e)}

    def run_bulk(self, items: List[Dict[str, str]], instruction: str) -> List[Dict[str, Any]]:
        """
        items: [{"question_id": "...", "context_with_question": "..."}, ...]
        """
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=BULK_WORKERS) as pool:
            futures = [
                pool.submit(self._call_one, it["question_id"], instruction, it["context_with_question"])
                for it in items
            ]
            for f in as_completed(futures):
                results.append(f.result())

        # (необязательно) стабилизируем порядок как во входе
        idx = {it["question_id"]: i for i, it in enumerate(items)}
        results.sort(key=lambda r: idx.get(r["question_id"], 10**9))
        return results