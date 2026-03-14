# rag_api/routes_bulk.py
from flask import Blueprint, request, jsonify, current_app
import logging
from typing import cast, TYPE_CHECKING
from rag.pipeline import RagPipeline

# Проверка типов для IDE
if TYPE_CHECKING:
    from app_factory import RagApp

api_bulk_bp = Blueprint("api_bulk", __name__, url_prefix="/api/rag")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ссылка на Flask клиента, чтобы вытаскивать конфиги типа current_mode
app = cast('RagApp', current_app)

@api_bulk_bp.route("/query/bulk", methods=["POST"])
def bulk_query():
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "cot")
    safe_prompt = bool(data.get("safe_prompt", True))
    questions = data.get("questions") or []

    instruction = RagPipeline.build_instructions(current_mode=mode, safe_prompt=safe_prompt)

    items: list[dict] = []
    errs: list[dict] = []

    for q in questions:
        qid = q.get("question_id")
        question = q.get("question")
        results = q.get("results") or []

        if not qid or not question:
            errs.append({"question_id": qid or "?", "answer": None, "error": "missing question_id or question"})
            continue

        context = _format_context_from_results(results)
        cwq = RagPipeline.build_context_with_question(context, question)
        items.append({"question_id": qid, "context_with_question": cwq})

    llm_bulk_client = app.bulk_openai_client  # singleton из app_factory
    llm_results = llm_bulk_client.run_bulk(items=items, instruction=instruction)

    # сохранить исходный порядок questions
    idx = {q.get("question_id"): i for i, q in enumerate(questions)}
    all_results = (llm_results + errs)
    all_results.sort(key=lambda r: idx.get(r.get("question_id"), 10**9))

    return jsonify({"mode": mode, "count": len(all_results), "results": all_results})


def _format_context_from_results(results: list[dict]) -> str:
    """
    Делает ровно тот же формат контекста, что и текущий retriever:
    ЧАНК >>>>
    [chunk #i, distance=..., meta=...]
    <text>
    <<<<

    ---
    """
    if not results:
        return ""

    results_sorted = sorted(results, key=lambda r: r.get("rank", 10**9))

    parts: list[str] = []
    for i, r in enumerate(results_sorted):
        dist = r.get("distance")
        meta = r.get("meta") or {}
        text = r.get("text", "")

        header = f"ЧАНК >>>>\n[chunk #{i+1}, distance={dist}, meta={meta}]"
        parts.append(header + "\n" + text + "\n<<<<")

    return "\n\n---\n\n".join(parts)


@api_bulk_bp.route("/retrieval/topk_bulk", methods=["POST"])
def retrieval_topk_bulk():
    """ Bulk retrieval only.
    POST /api/rag/retrieval/topk_bulk
      {
        "k": 5,
        include_text: true|false,
        "questions": [{"id":"q1","question":"..."}, ... ]
      }
    :return:
      {
        "k": 5,
        include_text: true|false,
        "results": [{"id":"q1","results":[{"rank":1,"distance":...,"meta":{...},"text":"..."}]}, ... ]
      }
    """
    data = request.get_json(silent=True) or {}
    k = int(data.get("k", 5))
    questions = data.get("questions", [])
    include_text = bool(data.get("include_text", False))

    if not isinstance(questions, list) or not questions:
        return jsonify({"error": "questions must be a non-empty list"}), 400

    pipeline = app.pipeline
    retriever = app.retriever  # ChromaRetriever

    try:
        logger.info(f"retrieving {k} results for {len(questions)} questions with topk_bulk query")
        return jsonify(
            retriever.retrieve_topk_bulk(
                pipeline=pipeline,
                questions=questions,
                k=k,
                include_text=include_text
            )
        ), 200
    except ValueError as e:
        logger.error(f"ValueError: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": repr(e)}), 500