import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _json_default(o: Any):
    # datetime → ISO (удобно для графиков)
    if isinstance(o, datetime):
        return o.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    raise TypeError(f"Not JSON serializable: {type(o)}")


class MetricsStore:
    """
    Пишем jsonl: одна метрика = одна строка JSON.
    На вход можно давать dict или dataclass.
    """
    def __init__(self, out_dir: str | Path):
        self._out_dir = Path(out_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def append_jsonl(self, filename: str, rows: list[Any]) -> None:
        path = self._out_dir / filename
        with path.open("a", encoding="utf-8") as f:
            for r in rows:
                obj = asdict(r) if is_dataclass(r) else r
                f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")