import json
from pathlib import Path
from domain.models import RunSummary

class RunSummaryWriter:
    """
    Пишет итог каждого run одной строкой в общий runs.jsonl (append).
    Формат: одна строка = один JSON-объект.
    """

    def __init__(self, runs_dir: Path, filename: str = "runs.jsonl") -> None:
        self._runs_dir = runs_dir
        self._file = runs_dir / filename

    def write(self, summary: RunSummary) -> Path:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": summary.status,
            "error": summary.error,
            "files": {
                "added": summary.added_files,
                "modified": summary.modified_files,
                "deleted": summary.deleted_files,
            },
            "chunks": {
                "prepared": summary.chunks_prepared,
                "db_upserted": summary.chunks_db_upserted,
                "db_deleted": summary.chunks_db_deleted,
            },
            "db_index": {
                "size_bytes": summary.index_size_bytes,
            },
            "started_at_ms": summary.started_at_ms,
            "finished_at_ms": summary.finished_at_ms,
            "duration_ms": (
                summary.duration_ms
                if summary.duration_ms is not None
                else None
            ),
        }

        # append одной строкой
        with self._file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

        return self._file