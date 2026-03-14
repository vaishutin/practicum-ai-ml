import httpx


class RagApiClient:
    def __init__(self, base_url: str, timeout_s: int = 120):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    def bulk_generate(self, *, mode: str, questions: list[dict], safe_prompt: bool = True) -> list[dict]:
        """
        questions: [
          {"question_id","question","k","results":[{"rank","distance","meta","text"}...]}
        ]
        returns: [{"question_id": "...", "answer": "...", ...}, ...]
        """
        payload = {"mode": mode, "safe_prompt": bool(safe_prompt), "questions": questions}
        resp = self._client.post("/api/rag/query/bulk", json=payload)
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("results") or []