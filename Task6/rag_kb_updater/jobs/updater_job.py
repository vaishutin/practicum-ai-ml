import hashlib
import logging
import time
from pathlib import Path

from domain.models import FileInfo, RunSummary
from services.manifest_store import ManifestStore
from services.diff_engine import DiffEngine
from services.chunker import Chunker
from services.index_builder import IndexBuilder
from services.run_summary_writer import RunSummaryWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_updater_job")

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
            update_strategy: str = "replace_file",
    ) -> None:
        self._kb_root = kb_root
        self._chunks_path = chunks_path
        self._manifest_store = manifest_store
        self._diff_engine = diff_engine
        self._chunker = chunker
        self._index_builder = index_builder
        self._summary_writer = summary_writer
        self._update_strategy = update_strategy

    def run(self) -> RunSummary:
        summary = RunSummary(started_at_ms=time.time_ns() // 1_000_000)
        logger.info("KB updater run started")

        try:
            old_manifest = self._manifest_store.load()
            new_manifest = self._scan_files(self._kb_root)

            files_diff = self._diff_engine.diff(old_manifest, new_manifest)
            summary.added_files = len(files_diff.added)
            summary.modified_files = len(files_diff.modified)
            summary.deleted_files = len(files_diff.deleted)

            logger.info(
                "Diff: added=%d modified=%d deleted=%d",
                summary.added_files, summary.modified_files, summary.deleted_files
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

            # 3) обновление manifest
            self._manifest_store.save(new_manifest)

            summary.status = "success"
            logger.info("KB updater run success")
            return summary

        except Exception as e:
            summary.status = "failed"
            summary.error = repr(e)
            logger.error("KB updater run failed", exc_info=True)
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