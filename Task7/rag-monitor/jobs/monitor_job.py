# jobs/monitor_job.py
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(
    r"(?is)(?:^|\n)\s*>>\s*Ответ\s*:\s*(.*?)\s*(?=(?:\n\s*>>\s*\w+[^:]*\s*:)|\Z)"
)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RagMonitorJob:
    """
    Один прогон:
      1) читает kb-updater/_runs/runs.jsonl
      2) берёт новые "хорошие" run_id
      3) читает raw retrieval log для run_id
      4) считает retrieval метрики и пишет в _metrics/retrieval_metrics.jsonl
      5) дергает rag-api-safe bulk генерацию, пишет техлог в _runs/generation_logs/<run_id>_answers.jsonl
      6) считает ragas метрики и пишет в _metrics/ragas_metrics.jsonl
      7) помечает run_id обработанным (processed_runs.jsonl)
    """

    def __init__(
            self,
            *,
            raw_logs_reader,
            metrics_store,
            run_summary_writer,
            rag_api_client,
            retrieval_metrics_evaluator,
            ragas_metrics_evaluator,
            golden_dir: str | Path,
            runs_dir: str | Path,
            mode: str = "few_shot",
            safe_prompt: bool = True,
            ragas_concurrency: int = 8,
    ) -> None:
        self._raw = raw_logs_reader
        self._store = metrics_store
        self._summary = run_summary_writer
        self._rag = rag_api_client
        self._retr_eval = retrieval_metrics_evaluator
        self._ragas_eval = ragas_metrics_evaluator

        self._golden_dir = Path(golden_dir)
        self._runs_dir = Path(runs_dir)
        self._gen_logs_dir = self._runs_dir / "generation_logs"
        self._gen_logs_dir.mkdir(parents=True, exist_ok=True)

        self._mode = mode
        self._safe_prompt = bool(safe_prompt)
        self._ragas_concurrency = int(ragas_concurrency)

    def run(self) -> None:
        logger.debug("RagMonitorJob started ==================")
        # 1) Try to find new unprocessed runs
        if not (valid_unprocessed_runs := self._try_find_valid_unprocessed_runs()):
            logger.info("No new runs to process - job is finished ")
            return
        logger.info("Found %d new runs to process", len(valid_unprocessed_runs))

        for run in valid_unprocessed_runs:
            run_id = run["run_id"]
            try:
                self._process_one_run(run)
                self._summary.mark_processed(run_id, retrieval=True, ragas=True, extra={"mode": self._mode})
            except Exception:
                logger.error("Failed processing run_id=%s", run_id, exc_info=True)

    # -----------------------
    # internals
    # -----------------------
    def _try_find_valid_unprocessed_runs(self) -> list[dict[str, Any]]:
        processed_ids = self._summary.read_processed_ids()
        logger.info("Found already processed runs from summary: %d ", len(processed_ids))

        runs = self._raw.read_runs()
        candidates = [
            r for r in runs
            if r.get("status") == "success"
               and r.get("index_status") == "INDEX_UPDATED"
               and r.get("golden_retrieval") is not None
        ]
        unprocessed = [r for r in candidates if (r.get("run_id") and r["run_id"] not in processed_ids)]
        logger.info("Found unprocessed runs: %d", len(unprocessed))

        valid_unprocessed = []
        for r in unprocessed:
            if r.get("golden_retrieval", {}).get("raw_log_path"):
                valid_unprocessed.append(r)
            else:
                logger.warning("run_id=%s has golden_retrieval but no raw_log_path", r["run_id"])
        return valid_unprocessed

    def _process_one_run(self, unprocessed_run: dict[str, Any]) -> None:
        run_id = unprocessed_run["run_id"]
        gr = unprocessed_run.get("golden_retrieval") or {}
        raw_log_path = gr.get("raw_log_path")
        logger.info(f"Starting processing run_id={raw_log_path}")

        # 1) Reading retrieval data
        retrieval_log_rows = self._raw.read_retrieval_log_by_path(raw_log_path)
        if not retrieval_log_rows:
            logger.warning("run_id=%s retrieval log is empty: %s", run_id, raw_log_path)
            return
        logger.info(f"Got retrieval golden set and retrieved log rows for run_id={run_id}: {len(retrieval_log_rows)}")

        # 2) Retrieval metrics processing
        self._process_retrieval_metrics(run_id, retrieval_log_rows)

        # 3) Answers generation
        generation_answers = self._generate_answers_for_retrieval(run_id, retrieval_log_rows)

        # 4) Ragas metrics processing
        self._process_ragas_metrics(run_id, retrieval_log_rows, generation_answers)

    def _process_retrieval_metrics(self, run_id: str, retrieval_log_rows: list[dict[str, Any]]) -> dict:
        # 1) Get retrieval_golden set (question + expecting_embedding_ids)
        golden_set_by_qid = self._load_retrieval_golden_set(self._golden_dir / "retrieval_golden.jsonl")

        k = int(retrieval_log_rows[0].get("k") or 0)
        self._retr_eval._golden = golden_set_by_qid
        retrieval_metrics = self._retr_eval.evaluate(retrieval_log_rows, k=k)
        self._store.append_jsonl(
            "retrieval_metrics.jsonl",
            [{"ts": _ts(), "run_id": run_id, "k": k, **retrieval_metrics["summary"]}],
        )
        self._store.append_jsonl(
            "retrieval_metrics_per_question.jsonl",
            [
                {"ts": _ts(), "run_id": run_id, "k": k, **row}
                for row in retrieval_metrics["per_question"]
            ],
        )
        logger.info("Retrieval metrics evaluated and stored, summary: %s", retrieval_metrics["summary"])
        retrieval_metrics_per_qid = retrieval_metrics["per_question"]
        for it in retrieval_metrics_per_qid:
            qid = it.get("question_id")
            if not qid:
                continue
            logger.info(
                "retrieval qid=%s, hit_at_k=%s, recall_at_k=%s, mrr_at_k=%s, found=%s, expected=%s, first_rank=%s, scanned=%s",
                qid,
                it.get("hit_at_k"),
                it.get("recall_at_k"),
                it.get("mrr_at_k"),
                it.get("found"),
                it.get("expected"),
                it.get("first_rank"),
                it.get("scanned"),
            )

    def _generate_answers_for_retrieval(self, run_id: str, retrieval_log_rows: list[dict[str, Any]]) -> list[dict]:
        # 1) check if answers already generated
        path = self._generation_log_path(run_id)

        if path.exists():
            logger.info(f"generation_log already exists, skip generation, run_id={run_id}, path={path}")
            return self._read_generation_log(run_id)

        # 2) If not, generate answers
        bulk_questions = [
            {
                "question_id": rr.get("question_id"),
                "question": rr.get("question"),
                "k": rr.get("k"),
                "results": rr.get("results"),
            }
            for rr in retrieval_log_rows
        ]
        logger.info(
            f"Getting answers for {len(bulk_questions)} golden set questions using rag-api-safe, run_id={run_id}")
        bulk_answers = self._rag.bulk_generate(mode=self._mode, safe_prompt=self._safe_prompt, questions=bulk_questions)

        self._write_generation_log(run_id, bulk_answers)
        logger.info(f"Generation_log with answers saved, run_id={run_id}")
        return bulk_answers

    def _process_ragas_metrics(self, run_id: str, retrieval_log_rows: list[dict[str, Any]],
                               generation_answers: list[dict]) -> None:
        # 1) Prepare input data for ragas
        logger.info(f"Building data rows for ragas (answer, question, context), run_id={run_id}")
        ragas_rows = self._build_ragas_rows(retrieval_log_rows, generation_answers)

        # 2) Ragas metrics evaluation
        logger.info(f"Starting evaluation of ragas metrics, run_id={run_id}, rows={len(ragas_rows)}")
        ragas_out = asyncio.run(self._ragas_eval.evaluate_batch(ragas_rows, concurrency=self._ragas_concurrency))

        # 3) Store metrics
        ragas_per_item = ragas_out.get("per_item") or []

        # summary
        self._store.append_jsonl(
            "ragas_metrics.jsonl",
            [{
                "ts": _ts(),
                "run_id": run_id,
                "mode": self._mode,
                "model": ragas_out.get("model"),
                "count": ragas_out.get("count"),
                "faithfulness_avg": self._round3(ragas_out.get("faithfulness_avg")),
                "answer_relevancy_avg": self._round3(ragas_out.get("answer_relevancy_avg")),
            }],
        )

        # per-question
        self._store.append_jsonl(
            "ragas_metrics_per_question.jsonl",
            [
                {
                    "ts": _ts(),
                    "run_id": run_id,
                    "mode": self._mode,
                    "model": ragas_out.get("model"),
                    "question_id": it.get("question_id"),
                    "faithfulness": self._round3(it.get("faithfulness")),
                    "answer_relevancy": self._round3(it.get("answer_relevancy")),
                }
                for it in ragas_per_item
                if it.get("question_id")
            ],
        )

        # 4) Logging
        logger.info(
            "ragas summary: run_id=%s, mode=%s, n=%s, faith=%.3f, answer_rel=%.3f",
            run_id, self._mode, ragas_out.get("count") or 0,
            self._round3(ragas_out.get("faithfulness_avg") or 0.0),
            self._round3(ragas_out.get("answer_relevancy_avg") or 0.0),
        )

        for it in ragas_per_item:
            qid = it.get("question_id")
            if not qid:
                continue
            logger.info(
                "ragas qid=%s, faith=%.3f, answer_rel=%.3f",
                qid,
                self._round3(it.get("faithfulness") or 0.0),
                self._round3(it.get("answer_relevancy") or 0.0),
            )

    @staticmethod
    def _round3(v):
        return round(v, 3) if isinstance(v, float) else v

    @staticmethod
    def _extract_answer_for_eval(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        m = _ANSWER_RE.search(text)
        return m.group(1).strip() if m else text.strip()

    def _build_ragas_rows(self, retrieval_rows: list[dict], bulk_answers: list[dict]) -> list[dict]:
        # полный ответ, который может содержать cot, заголовки, фрагменты текста (в зависимости от режима)
        full_formatted_ans = {a.get("question_id"): a.get("answer") for a in bulk_answers if a.get("answer")}

        rows: list[dict] = []
        for rr in retrieval_rows:
            qid = rr.get("question_id")
            question = rr.get("question")
            answer = self._extract_answer_for_eval(full_formatted_ans.get(qid, ""))

            if not qid or not question or not answer:
                continue

            results = rr.get("results") or []
            results_sorted = sorted(results, key=lambda r: r.get("rank", 10 ** 9))
            contexts = [r.get("text") for r in results_sorted if r.get("text")]

            # # попытка ограничить контекст - вроде не помогает улучшить faithfulness
            # logger.info(f"Ограничиваем contexts размером={len(contexts)} до 3 перед входом в ragas")
            # contexts = contexts[:3]

            rows.append({"question_id": qid, "question": question, "answer": answer, "contexts": contexts})
        return rows

    def _generation_log_path(self, run_id: str) -> Path:
        return self._gen_logs_dir / f"{run_id}_answers.jsonl"

    def _read_generation_log(self, run_id: str) -> list[dict]:
        path = self._generation_log_path(run_id)
        answers: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                answers.append(json.loads(line))
        return answers

    def _write_generation_log(self, run_id: str, answers: list[dict]) -> None:
        path = self._gen_logs_dir / f"{run_id}_answers.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for a in answers:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    def _load_retrieval_golden_set(self, path: Path) -> dict[str, Any]:
        """
        retrieval_golden.jsonl:
          {"question_id":"q9","expected_section_paths":["[['Перехватчик BABOLAT','Описание']]", ...]}

        Возвращаем dict[qid] -> GoldenExpected (из retrieval_metrics_evaluator.py).
        """
        # lazy import, чтобы не было циклов
        from eval.retrieval_metrics_evaluator import GoldenExpected

        out: dict[str, Any] = {}
        if not path.exists():
            logger.warning("Golden file not found: %s", path)
            return out

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                qid = o.get("question_id")
                exp = o.get("expected_embedding_ids") or []
                if qid:
                    out[qid] = GoldenExpected(question_id=qid, expected_embedding_ids=list(exp))

        logger.info(f"Loaded golden questions set from {self._golden_dir}/retrieval_golden.jsonl")
        return out
