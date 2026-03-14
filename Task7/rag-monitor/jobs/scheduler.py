import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from jobs.monitor_job import RagMonitorJob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, job: RagMonitorJob, cron_expr: str, job_name: str) -> None:
        self._job = job
        self._cron_expr = cron_expr
        self._sched = BackgroundScheduler()
        self._job_name = job_name

    def start(self, run_on_start: bool) -> None:
        logger.info("Starting APScheduler cron='%s'", self._cron_expr)

        trigger = CronTrigger.from_crontab(self._cron_expr)
        self._sched.add_job(
            func=self._safe_run,
            trigger=trigger,
            id=self._job_name,
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self._sched.start()

        if run_on_start:
            self._safe_run()

    def shutdown(self) -> None:
        self._sched.shutdown(wait=True)

    def _safe_run(self) -> None:
        try:
            self._job.run()
        except Exception:
            logger.error("Scheduler job execution failed", exc_info=True)