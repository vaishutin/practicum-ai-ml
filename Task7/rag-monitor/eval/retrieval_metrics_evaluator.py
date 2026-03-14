from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class GoldenExpected:
    question_id: str
    expected_embedding_ids: list[str]  # напр. ["13::chunk_11", "13::chunk_10"]


def _get_embedding_id(result: dict) -> Optional[str]:
    """
    Достаём embedding_id из result.
    """
    eid = result.get("embedding_id")
    return eid if isinstance(eid, str) and eid else None


class RetrievalMetricsEvaluator:
    def __init__(self, golden_by_qid: dict[str, GoldenExpected]) -> None:
        self._golden = golden_by_qid

    def evaluate(self, rows: Iterable[dict], k: int) -> dict:
        per_q = [m for m in (self._eval_one(row, k) for row in rows) if m is not None]
        return {"summary": self._summarize(per_q), "per_question": per_q}

    def _eval_one(self, row: dict, k: int) -> Optional[dict]:
        qid = row.get("question_id")
        g = self._golden.get(qid) if isinstance(qid, str) else None
        if not qid or not g:
            return None

        expected = [e for e in g.expected_embedding_ids if isinstance(e, str) and e]
        if not expected:
            return None

        results = row.get("results") or []
        found_ids, first_rank, scanned = self._scan_topk(results, k, set(expected))

        found = sum(1 for e in expected if e in found_ids)
        hit = 1.0 if found > 0 else 0.0
        recall = found / len(expected)
        mrr = (1.0 / first_rank) if first_rank else 0.0

        return {
            "question_id": qid,
            "hit_at_k": round(hit, 3),
            "recall_at_k": round(recall, 3),
            "mrr_at_k": round(mrr, 3),
            # полезно для дебага
            "found": found,
            "expected": len(expected),
            "first_rank": first_rank,
            "scanned": scanned,  # сколько результатов реально просмотрели до break
        }

    def _scan_topk(self, results: list[dict], k: int, expected_set: set[str]) -> tuple[set[str], Optional[int], int]:
        """
        Считаем, что results уже отсортированы по rank
        Если rank > k — выходим (break).
        """
        found_ids: set[str] = set()
        first_rank: Optional[int] = None
        scanned = 0

        for r in results:
            rk = r.get("rank")
            if isinstance(rk, int) and rk > k:
                break

            scanned += 1
            eid = _get_embedding_id(r)
            if not eid:
                continue

            found_ids.add(eid)
            if first_rank is None and eid in expected_set and isinstance(rk, int):
                first_rank = rk

        return found_ids, first_rank, scanned

    @staticmethod
    def _summarize(per_q: list[dict]) -> dict:
        n = len(per_q)
        if n == 0:
            return {"count": 0, "hit_at_k": None, "recall_at_k": None, "mrr_at_k": None}

        def avg(key: str) -> float:
            return round(sum(r[key] for r in per_q) / n, 3)

        return {
            "count": n,
            "hit_at_k": avg("hit_at_k"),
            "recall_at_k": avg("recall_at_k"),
            "mrr_at_k": avg("mrr_at_k"),
        }
