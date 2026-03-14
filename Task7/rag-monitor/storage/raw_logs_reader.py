import json
from pathlib import Path
from typing import Any, Iterable


class RawLogsReader:
    """
    Читает логи rag-kb-updater:
      - runs.jsonl
      - retrieval_logs/<run_id>_raw_retrieval.jsonl

    Важно:
      - section_path в meta у вас хранится как СТРОКА, которая выглядит как python-list.
        Мы её не парсим (нам достаточно сравнения подстрокой).
      - работаем с dict'ами, без тяжёлых моделей.
    """
    def __init__(self, kb_runs_dir: str | Path):
        self._kb_runs_dir = Path(kb_runs_dir)

    def iter_jsonl(self, path: str | Path) -> Iterable[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        def gen():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
        return gen()

    def read_runs(self) -> list[dict[str, Any]]:
        runs_path = self._kb_runs_dir / "runs.jsonl"
        return list(self.iter_jsonl(runs_path))

    def read_retrieval_log_by_path(self, raw_log_path: str) -> list[dict[str, Any]]:
        # raw_log_path приходит из runs.jsonl (например "_runs/retrieval_logs/...jsonl")
        # Он относительный к корню kb-updater. У нас kb_runs_dir = kb-updater/_runs,
        # значит "_runs/..." надо корректно резолвить.
        p = Path(raw_log_path)

        # 1) если путь абсолютный — читаем как есть
        if p.is_absolute():
            return list(self.iter_jsonl(p))

        # 2) если начинается с "_runs/..." — отрезаем "_runs" и резолвим от kb_runs_dir
        parts = p.parts
        if parts and parts[0] == "_runs":
            p = self._kb_runs_dir / Path(*parts[1:])
        else:
            # 3) иначе считаем, что путь относительный от kb_runs_dir
            p = self._kb_runs_dir / p

        return list(self.iter_jsonl(p))