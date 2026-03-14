import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from domain.models import FileInfo, RunSummary
from services.manifest_store import ManifestStore
from services.diff_engine import DiffEngine
from services.chunker import Chunker
from services.index_builder import IndexBuilder
from services.run_summary_writer import RunSummaryWriter
from services.golden_retrieval_runner import GoldenRetrievalRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class UpdaterJob:
    def __init__(
            self,
            kb_root: Path,
            chunks_path: Path,
            manifest_store: ManifestStore,
            diff_engine: DiffEngine,
            chunker: Chunker,
            index_builder: IndexBuilder,
            summary_writer: RunSummaryWriter,
            golden_runner: GoldenRetrievalRunner,
            golden_k: int = 5,
            update_strategy: str = "replace_file",
    ) -> None:
        self._kb_root = kb_root
        self._chunks_path = chunks_path
        self._manifest_store = manifest_store
        self._diff_engine = diff_engine
        self._chunker = chunker
        self._index_builder = index_builder
        self._summary_writer = summary_writer

        self._golden_runner = golden_runner
        self._golden_k = golden_k

        self._update_strategy = update_strategy

    def run(self) -> RunSummary:
        run_id = self._make_run_id()
        summary = RunSummary(run_id=run_id, started_at_ms=time.time_ns() // 1_000_000)
        logger.info("KB updater run started: run_id=%s", run_id)

        try:
            old_manifest = self._manifest_store.load()
            new_manifest = self._scan_files(self._kb_root)

            files_diff = self._diff_engine.diff(old_manifest, new_manifest)
            summary.added_files = len(files_diff.added)
            summary.modified_files = len(files_diff.modified)
            summary.deleted_files = len(files_diff.deleted)

            summary.index_status = (
                "INDEX_SAME"
                if (summary.added_files + summary.modified_files + summary.deleted_files) == 0
                else "INDEX_UPDATED"
            )

            logger.info(
                "Diff: added=%d modified=%d deleted=%d index_status=%s",
                summary.added_files, summary.modified_files, summary.deleted_files, summary.index_status
            )

            if self._update_strategy != "replace_file":
                raise ValueError(f"Unsupported UPDATE_STRATEGY={self._update_strategy}")

            # 1) подготовка чанков - запись jsonl (переписывается полностью)
            chunks_prepared_count = self._chunker.prepare_chunks()
            summary.chunks_prepared = chunks_prepared_count

            # 2) индекс: удалить все modified+deleted
            # можно удалять только deleted, потому что вроде есть upsert в chroma (но не проверял)
            to_delete = files_diff.deleted + files_diff.modified
            if to_delete:
                summary.chunks_db_deleted = self._index_builder.delete_by_source_paths(to_delete)

            # 3) индекс: upsert все added+modified (плюс fallback на add)
            changed_files = set(files_diff.added + files_diff.modified)
            summary.chunks_db_upserted = self._index_builder.build_index(changed_files)
            summary.index_size_bytes = self._index_builder.index_size_bytes()

            summary.status = "success"
            logger.info("KB index updater run finished successfully: run_id=%s", run_id)

            # 4) golden retrieval — только если индекс реально менялся
            if summary.status == "success" and summary.index_status == "INDEX_UPDATED":
            # if summary.status == "success":
                logger.info("Starting golden retrieval")
                gr = self._golden_runner.run(run_id=run_id, k=self._golden_k)
                summary.golden_retrieval = {
                    "k": gr.k,
                    "questions": gr.questions,
                    "raw_log_path": gr.raw_log_path,
                }
                logger.info("Golden retrieval written: %s", gr.raw_log_path)
            else:
                logger.info("Golden retrieval skipped (INDEX_SAME)")

            # 3) Save manifest if only everything went well
            self._manifest_store.save(new_manifest)
            logger.info(f"Saving manifest for {run_id}, status={summary.status}, index_status={summary.index_status}")
            return summary

        except Exception as e:
            summary.status = "failed"
            summary.error = repr(e)
            logger.error(
                f"Updater job index updater run failed, status={summary.status}, manifest didnt updated: index_status={summary.index_status}, run_id={run_id}",
                exc_info=True)
            return summary

        finally:
            summary.finished_at_ms = time.time_ns() // 1_000_000
            summary.duration_ms = summary.finished_at_ms - summary.started_at_ms
            out = self._summary_writer.write(summary)
            logger.info("Run summary written: %s", out)

    def _scan_files(self, kb_root: Path) -> dict[str, FileInfo]:
        out: dict[str, FileInfo] = {}
        for p in kb_root.rglob("*.md"):
            rel = str(p.relative_to(kb_root))
            st = p.stat()
            sha = self._sha256(p)
            out[rel] = FileInfo(path=rel, size=st.st_size, mtime=st.st_mtime, sha256=sha)
        return out

    def _sha256(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _make_run_id() -> str:
        # пример: 2026-01-05T02-10-00Z
        dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H-%M-%SZ")