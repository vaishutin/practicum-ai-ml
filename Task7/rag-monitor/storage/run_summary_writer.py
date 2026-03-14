import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunSummaryWriter:
    """
    Храним, какие run_id уже обработаны.
    Формат jsonl, чтобы было легко дописывать и смотреть диффом.

    processed_runs.jsonl:
      {"run_id": "...", "processed_at": "2026-01-11T...", "retrieval": true, "ragas": true}
    """
    def __init__(self, runs_dir: str | Path):
        self._runs_dir = Path(runs_dir)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._runs_dir / "processed_runs.jsonl"

    def read_processed_ids(self) -> set[str]:
        if not self._path.exists():
            return set()

        out: set[str] = set()
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.add(json.loads(line).get("run_id"))
                except Exception:
                    # если кто-то руками испортил строку — не падаем
                    continue
        return {x for x in out if x}

    def mark_processed(self, run_id: str, *, retrieval: bool, ragas: bool, extra: dict[str, Any] | None = None) -> None:
        rec = {"run_id": run_id, "processed_at": _ts(), "retrieval": bool(retrieval), "ragas": bool(ragas)}
        if extra:
            rec.update(extra)

        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Summary for processed run_id=%s is saved", run_id)
