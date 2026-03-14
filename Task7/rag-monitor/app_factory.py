import logging

from clients.rag_api_client import RagApiClient
from config import Config
from eval.ragas_metrics_evaluator import RagasMetricsEvaluator
from eval.retrieval_metrics_evaluator import RetrievalMetricsEvaluator
from storage.metrics_store import MetricsStore
from storage.raw_logs_reader import RawLogsReader
from storage.run_summary_writer import RunSummaryWriter
from jobs.monitor_job import RagMonitorJob
from jobs.scheduler import Scheduler

logger = logging.getLogger(__name__)


class App:
    def __init__(self, scheduler: Scheduler, run_on_start: bool) -> None:
        self._scheduler = scheduler
        self._run_on_start = run_on_start

    def start(self) -> None:
        self._scheduler.start(run_on_start=self._run_on_start)


def create_app() -> App:
    cfg = Config()
    cfg.ensure_dirs()

    # -------- IO --------
    raw_logs_reader = RawLogsReader(cfg.KB_RUNS_DIR)
    metrics_store = MetricsStore(cfg.METRICS_DIR)
    run_summary_writer = RunSummaryWriter(cfg.RUNS_DIR)

    # -------- Clients --------
    rag_api_client = RagApiClient(
        base_url=cfg.RAG_API_BASE_URL,
        timeout_s=cfg.RAG_API_TIMEOUT_S,
    )

    # -------- Evaluators --------
    # golden подгружается внутри monitor_job
    retrieval_evaluator = RetrievalMetricsEvaluator(golden_by_qid={})

    ragas_evaluator = RagasMetricsEvaluator(
        model=cfg.RAGAS_JUDGE_MODEL,
        emb_model=cfg.RAGAS_EMB_MODEL,
    )

    # -------- Job --------
    job = RagMonitorJob(
        raw_logs_reader=raw_logs_reader,
        metrics_store=metrics_store,
        run_summary_writer=run_summary_writer,
        rag_api_client=rag_api_client,
        retrieval_metrics_evaluator=retrieval_evaluator,
        ragas_metrics_evaluator=ragas_evaluator,
        golden_dir=cfg.GOLDEN_DIR,
        runs_dir=cfg.RUNS_DIR,
        mode=cfg.MODE,
        safe_prompt=cfg.SAFE_PROMPT,
        ragas_concurrency=cfg.RAGAS_CONCURRENCY,
    )

    # -------- Scheduler --------
    scheduler = Scheduler(
        job=job,
        cron_expr=cfg.CRON,
        job_name="rag-monitor",
    )

    logger.info(
        "RAG Monitor config: KB_RUNS_DIR=%s | MODE=%s | SAFE_PROMPT=%s | CRON=%s | RUN_ON_START=%s | RAG_API=%s",
        cfg.KB_RUNS_DIR,
        cfg.MODE,
        cfg.SAFE_PROMPT,
        cfg.CRON,
        cfg.RUN_ON_START,
        cfg.RAG_API_BASE_URL,
    )

    return App(scheduler, run_on_start=cfg.RUN_ON_START)