import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "y", "on")


@dataclass(frozen=True)
class Config:
    # где лежат логи kb-updater (каталог _runs)
    KB_RUNS_DIR: Path = Path(os.getenv("KB_RUNS_DIR", "../rag-kb-updater/_runs"))

    # наши директории
    GOLDEN_DIR: Path = Path(os.getenv("GOLDEN_DIR", "./_golden"))
    METRICS_DIR: Path = Path(os.getenv("METRICS_DIR", "./_metrics"))
    RUNS_DIR: Path = Path(os.getenv("RUNS_DIR", "./_runs"))

    # rag-api-safe
    RAG_API_BASE_URL: str = os.getenv("RAG_API_BASE_URL", "http://localhost:8000")
    MODE: str = os.getenv("MODE", "few_shot")
    SAFE_PROMPT: bool = _env_bool("SAFE_PROMPT", True)

    # cron/scheduler
    CRON: str = os.getenv("CRON", "*/5 * * * *")  # раз в 5 минут
    RUN_ON_START: bool = _env_bool("RUN_ON_START", True)

    # ragas
    RAGAS_JUDGE_MODEL: str = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
    RAGAS_EMB_MODEL: str = os.getenv("RAGAS_EMB_MODEL", "text-embedding-3-small")
    RAGAS_CONCURRENCY: int = int(os.getenv("RAGAS_CONCURRENCY", "8"))

    # http
    RAG_API_TIMEOUT_S: int = int(os.getenv("RAG_API_TIMEOUT_S", "120"))

    def ensure_dirs(self) -> None:
        self.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self.RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (self.RUNS_DIR / "generation_logs").mkdir(parents=True, exist_ok=True)