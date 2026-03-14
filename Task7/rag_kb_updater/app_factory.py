import logging

from config import Config
from services.manifest_store import ManifestStore
from services.diff_engine import DiffEngine
from services.chunker import Chunker
from services.index_builder import IndexBuilder
from services.run_summary_writer import RunSummaryWriter
from jobs.updater_job import UpdaterJob
from services.scheduler import Scheduler

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from pathlib import Path
from services.rag_api_safe_client import RagApiSafeClient
from services.golden_retrieval_runner import GoldenRetrievalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class UpdaterApp:
    def __init__(self, scheduler: Scheduler, job: UpdaterJob, cfg: Config) -> None:
        self._scheduler = scheduler
        self._job = job
        self._cfg = cfg

    def start(self) -> None:
        if not self._cfg.sched_enabled:
            logger.info("SCHED_ENABLED=false -> run once and exit")
            self._job.run()
            return

        self._scheduler.start(run_on_start=self._cfg.run_on_start)

        # держим процесс живым (для Docker)
        try:
            import time
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            self._scheduler.shutdown()

def create_app(config: Config | None = None) -> UpdaterApp:
    cfg = config or Config.from_env()
    logger.info(f"RUNS_DIR: {cfg.runs_dir}")

    manifest_store = ManifestStore(cfg.manifest_path)
    diff_engine = DiffEngine()

    logger.info(f"Создаем токенизатор для модели: {cfg.embedder_model}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.embedder_model)
    logger.info("Создаем чанкер")
    chunker = Chunker(tokenizer, cfg.embedder_model, cfg.max_tokens, cfg.overlap_tokens)

    # 3. Эмбеддер (модель эмбеддингов)
    logger.info(f"Создаем эмбеддер (модель эмбеддингов): {cfg.embedder_model}")
    embedder = SentenceTransformer(cfg.embedder_model)

    # 4. Клиент сhroma
    if cfg.chroma_mode == "http":
        logger.info(f"Создается http-клиент Chroma для: {cfg.chroma_host}:{cfg.chroma_port}")
        chroma_client = chromadb.HttpClient(
            host=cfg.chroma_host,
            port=cfg.chroma_port,
            settings=Settings(anonymized_telemetry=False),
        )
    else:
        logger.info(f"Создается persistent-клиент Chroma для хранилища: {cfg.chroma_dir}")
        chroma_client = chromadb.PersistentClient(path=cfg.chroma_dir)

    logger.info(f"Создается IndexBuilder для хранилища: {cfg.chroma_dir}")
    index_builder = IndexBuilder(
        embedder=embedder,
        chroma_client=chroma_client,
        chunks_path=cfg.chunks_path,
        chroma_dir=cfg.chroma_dir,
        chroma_collection_name=cfg.chroma_collection,
        embedder_model=cfg.embedder_model,
        max_tokens=cfg.max_tokens,
        overlap_tokens=cfg.overlap_tokens,
    )

    summary_writer = RunSummaryWriter(cfg.runs_dir)

    rag_client = RagApiSafeClient(base_url=cfg.rag_api_safe_url, timeout_s=60.0)
    golden_runner = GoldenRetrievalRunner(
        questions_path=Path(cfg.golden_questions_path),
        logs_dir=Path(cfg.retrieval_logs_dir),
        client=rag_client,
    )

    job = UpdaterJob(
            kb_root=cfg.kb_root,
            chunks_path=cfg.chunks_path,
            manifest_store=manifest_store,
            diff_engine=diff_engine,
            chunker=chunker,
            index_builder=index_builder,
            summary_writer=summary_writer,
            golden_runner=golden_runner,
            golden_k=cfg.golden_top_k,
            update_strategy=cfg.update_strategy,
        )

    scheduler = Scheduler(job=job, cron_expr=cfg.sched_cron)
    return UpdaterApp(scheduler=scheduler, job=job, cfg=cfg)