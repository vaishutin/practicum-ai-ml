from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int
    mtime: float
    sha256: str

@dataclass(frozen=True)
class Diff:
    added: list[str]
    modified: list[str]
    deleted: list[str]

@dataclass
class RunSummary:
    started_at_ms: int
    finished_at_ms: int | None = None
    duration_ms: int | None = None
    status: str = "running"        # running/success/failed
    error: str | None = None

    added_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0

    chunks_prepared: int = 0
    chunks_db_upserted: int = 0
    chunks_db_deleted: int = 0

    index_size_bytes: int | None = None