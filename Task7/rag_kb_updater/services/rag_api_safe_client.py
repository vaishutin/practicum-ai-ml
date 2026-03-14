from __future__ import annotations

import httpx


class RagApiSafeClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_s

    def retrieval_topk_bulk(self, *, k: int, questions: list[dict[str, str]]) -> dict:
        """
        POST /api/rag/retrieval/topk_bulk
        Возвращает JSON как dict:
          {"k": k, "results": [{"id": "...", "results":[...]}]}
        """
        url = f"{self._base_url}/api/rag/retrieval/topk_bulk"
        payload = {"k": k, "include_text": True, "questions": questions}

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"rag-api-safe HTTP {e.response.status_code}: {e.response.text}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"rag-api-safe request failed: {e!r}") from e