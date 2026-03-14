from dataclasses import dataclass
from pathlib import Path
import os

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default

def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).lower() in ("1", "true", "yes", "y", "on")

@dataclass(frozen=True)
class Config:
    kb_root: Path
    manifest_path: Path
    runs_dir: Path

    chroma_dir: Path
    chroma_collection: str
    embedder_model: str
    chunks_path: Path

    chroma_mode: str
    chroma_host: str
    chroma_port: int

    max_tokens: int
    overlap_tokens: int

    sched_enabled: bool
    sched_cron: str
    run_on_start: bool

    update_strategy: str  # "replace_file" for MVP

    @staticmethod
    def from_env() -> "Config":
        return Config(
            kb_root=Path(_env("KB_ROOT", "../knowledge_base")),
            manifest_path=Path(_env("MANIFEST_PATH", "../artifacts/manifest.json")),
            runs_dir=Path(_env("RUNS_DIR", "../runs")),

            chroma_dir=Path(_env("CHROMA_DB_DIR", "../chroma_db")),
            chroma_collection=_env("CHROMA_COLLECTION", "kb"),
            embedder_model=_env("EMBEDDER_MODEL", "BAAI/bge-m3"),
            chunks_path=Path(_env("CHUNKS_PATH", "../artifacts/prepared_chunks/chunks.jsonl")),

            # http/persistent
            chroma_mode=_env("CHROMA_MODE", "persistent"),
            chroma_host=_env("CHROMA_HOST", "chroma"),
            chroma_port=int(_env("CHROMA_PORT", "8003")),

            max_tokens=int(_env("MAX_TOKENS", "256")),
            overlap_tokens=int(_env("OVERLAP_TOKENS", "64")),

            sched_enabled=_env_bool("SCHED_ENABLED", "true"),
            sched_cron=_env("SCHED_CRON", "*/1 * * * *"),
            run_on_start=_env_bool("SCHED_RUN_ON_START", "true"),
            update_strategy=_env("UPDATE_STRATEGY", "replace_file"),
        )