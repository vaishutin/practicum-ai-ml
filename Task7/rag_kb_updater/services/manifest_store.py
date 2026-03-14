import json
from pathlib import Path
from domain.models import FileInfo

class ManifestStore:
    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path

    def load(self) -> dict[str, FileInfo]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text(encoding="utf-8"))
        out: dict[str, FileInfo] = {}
        for p, fi in data.items():
            out[p] = FileInfo(path=p, size=fi["size"], mtime=fi["mtime"], sha256=fi["sha256"])
        return out

    def save(self, manifest: dict[str, FileInfo]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {p: {"size": fi.size, "mtime": fi.mtime, "sha256": fi.sha256} for p, fi in manifest.items()}
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")