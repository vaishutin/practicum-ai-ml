from flask import Blueprint, Response, request, jsonify

import json
from rag.config import INSTRUCTIONS, MODES
from flask import current_app
from typing import cast, TYPE_CHECKING

# Проверка типов для IDE
if TYPE_CHECKING:
    from app_factory import RagApp

import logging

JSON_UTF8 = "application/json; charset=utf-8"

api_bp = Blueprint("api", __name__, url_prefix="/api/rag")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ссылка на Flask клиента, чтобы вытаскивать конфиги типа current_mode
app = cast('RagApp', current_app)

# ------------------------
# HTTP ручки
# ------------------------
@api_bp.route("/query", methods=["POST"])
def rag_query():
    """
    Основная ручка: обработка запроса к RAG + LLM.
    JSON: { "question": "..." }
    Опционально "mode": "zero_shot|few_shot|cot" для override,
    иначе используется текущий выбранный режим.
    """
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question")
    override_mode = data.get("mode")
    logger.info("got POST /api/rag/query, mode: %s, question:  \"%s\"", override_mode, question)

    if not question:
        payload = {"error": "field 'question' is required"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=400,
        )
    mode = override_mode or app.pipeline.current_mode

    try:
        result = app.pipeline.run_rag_pipeline(question,
                                               mode,
                                               app.pipeline.safe_prompt)
        # result сам по себе dict вида:
        # {
        #   "answer": "...",
        #   "mode": "...",
        #   "used_top_k": RAG_TOP_K,
        # }

        logger.info("Sending back response to POST /api/rag/query, mode: %s, question:  \"%s\"", override_mode, question)
        return Response(
            json.dumps(result, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=200,
        )

    except Exception as e:
        payload = {"error": str(e)}
        logger.error("Got error: %s", str(e))

        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=500,
        )

@api_bp.route("/mode", methods=["GET", "POST"])
def rag_mode():
    """
    GET  /api/rag/mode  -> текущий режим
    POST /api/rag/mode  { "mode": "zero_shot|few_shot|cot" } -> смена режима
    """

    if request.method == "GET":
        logger.info("got GET /api/rag/mode, current_mode= %s", app.pipeline.current_mode)
        return jsonify({"current_mode": app.pipeline.current_mode, "available_modes": MODES})

    # POST
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode")
    logger.info("got POST /api/rag/mode, new mode= %s", mode)
    if mode not in MODES:
        logger.warning("mode: %s is not allowed by MODES list (%s)", mode, str(MODES))
        return jsonify({"error": f"mode must be one of {MODES}"}), 400

    app.pipeline.current_mode = mode
    logger.info("Mode is changed to: %s", app.pipeline.current_mode)
    return jsonify({"current_mode": app.pipeline.current_mode})

@api_bp.route("/toggle-safety", methods=["POST"])
def safe_prompt_mode():
    """
    POST /api/rag/toggle-safety
        {
            "safe_prompt": true|false,
            "retriever_chunks_filter": true|false,
            "retriever_text_cleaner": true|false
        }
    :return:
        {
            "safe_prompt": true|false,
            "retriever_chunks_filter": true|false,
            "retriever_text_cleaner": true|false
        }
    """

    data = request.get_json(force=True, silent=True) or {}
    new_safe_prompt = data.get("safe_prompt")
    new_retriever_chunks_filter = data.get("retriever_chunks_filter")
    new_retriever_text_cleaner = data.get("retriever_text_cleaner")
    base_message = f"got POST /api/rag/toggle-safety, body:{data}"
    if not (isinstance(new_safe_prompt, bool)
            and isinstance(new_retriever_chunks_filter, bool)
            and isinstance(new_retriever_chunks_filter, bool)):
        logger.error(
            f"{base_message} with invalid types")
        return jsonify({
            "error": "all fields must be a boolean (true or false)"
        }), 400


    app.pipeline.safe_prompt = new_safe_prompt
    app.pipeline.retriever_chunks_filter = new_retriever_chunks_filter
    app.pipeline.retriever_text_cleaner = new_retriever_text_cleaner
    logger.info(
        f"{base_message}, safety params is set to: safe_prompt=%s, retriever_chunks_filter=%s, retriever_text_cleaner=%s",
        new_safe_prompt, new_retriever_chunks_filter, new_retriever_text_cleaner)
    return jsonify({"safe_prompt": new_safe_prompt,
                    "retriever_chunks_filter": new_retriever_chunks_filter,
                    "retriever_text_cleaner": new_retriever_text_cleaner
                    })


@api_bp.route("/prompt", methods=["GET"])
def get_prompt():
    """
    GET /api/rag/prompt?mode=zero_shot|few_shot|cot
    Если mode не указан — отдаём промпт(инструкцию) текущего режима.
    """
    mode = request.args.get("mode")

    effective_mode = mode or app.pipeline.current_mode
    if effective_mode not in INSTRUCTIONS:
        payload = {"error": f"mode must be one of {MODES}"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=400
        )

    payload = {
        "mode": effective_mode,
        "prompt": INSTRUCTIONS[effective_mode],
    }

    return Response(
        json.dumps(payload, ensure_ascii=False),
        content_type=JSON_UTF8,
        status=200
    )


@api_bp.route("/prompt", methods=["PUT"])
def update_prompt():
    """
    PUT /api/rag/prompt
    JSON: { "mode": "zero_shot|few_shot|cot", "prompt": "новый текст шаблона" }
    """
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode")
    instruction_text = data.get("prompt")

    if mode not in MODES:
        payload = {"error": f"mode must be one of {MODES}"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=400
        )
    if not instruction_text:
        payload = {"error": "field 'prompt' is required"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type=JSON_UTF8,
            status=400
        )

    INSTRUCTIONS[mode] = instruction_text
    payload = {
        "mode": mode,
        "prompt": INSTRUCTIONS[mode],
    }

    return Response(
        json.dumps(payload, ensure_ascii=False),
        content_type=JSON_UTF8,
        status=200
    )