import logging
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from jobs.updater_job import UpdaterJob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("kb-scheduler")

class Scheduler:
    def __init__(self, job: UpdaterJob, cron_expr: str) -> None:
        self._job = job
        self._cron_expr = cron_expr
        self._sched = BackgroundScheduler()

    def start(self, run_on_start: bool) -> None:
        logger.info("Starting APScheduler cron='%s'", self._cron_expr)

        trigger = CronTrigger.from_crontab(self._cron_expr)
        self._sched.add_job(
            func=self._safe_run,
            trigger=trigger,
            id="kb_update_job",
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