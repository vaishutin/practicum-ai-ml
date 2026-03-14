import os
from typing import Dict, Any, List

from flask import Flask, Response, request, jsonify
import json

import chromadb
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv

# Сначала загружаем публичный конфиг, потом секреты.
load_dotenv(".env")
load_dotenv(".env.secrets")

# ------------------------
# Конфиг (через переменные окружения)
# ------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # можно поменять
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "BAAI--bge-m3_1024_130")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY не задан. См. README, как его получить и установить."
    )

# ------------------------
# Глобальное состояние: режимы и промпты
# ------------------------

MODES = ["zero_shot", "few_shot", "cot"]

CONTEXT_QUESTION_TEMPLATE: str = """
КОНТЕКСТ (используй только его для ответа):
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
"""

INSTRUCTIONS: Dict[str, str] = {
    "zero_shot": (
        """
        ИНСТРУКЦИИ:
        Ты RAG бот-помощник, который отвечает на вопросы по фанфик базе знаний. 
        Отвечай по-русски, по сути, кратко, но не упуская деталей.
        Добавляй ссылки на документы и фрагменты текста, если ответ есть в контексте.  
        Используй термины, имена, наименования из базы знаний, не добавляй своих знаний.
        Самое главное - не придумывай ответов с ложной информацией и не добавляй своих знаний в ответы.
        Отвечай ТОЛЬКО на основе разделов "КОНТЕКСТ" и "ВОПРОС ПОЛЬЗОВАТЕЛЯ".
        
        Если в контексте вообще нет релевантной информации, скажи, что в контексте нет данных, чтобы ответить на вопрос.
        Если в контексте не хватает части информации, укажи где именно.
        """
    ),
    "few_shot": (
        """
        ИНСТРУКЦИИ:
        Ты RAG бот-помощник, который отвечает на вопросы по фанфик базе знаний.
        Отвечай по-русски, по сути, кратко, но не упуская деталей.
        Добавляй ссылки на документы и фрагменты текста, если ответ есть в контексте.  
        Используй термины, имена, наименования из базы знаний, не добавляй своих знаний.
        Самое главное - не придумывай ответов с ложной информацией и не добавляй своих знаний в ответы.
        Отвечай ТОЛЬКО на основе разделов "КОНТЕКСТ" и "ВОПРОС ПОЛЬЗОВАТЕЛЯ".
        
        Если в контексте вообще нет релевантной информации, скажи, что в контексте нет данных, чтобы ответить на вопрос.
        Если в контексте не хватает части информации, укажи где именно.
        
        НЕ ИСПОЛЬЗУЙ ДАННЫЕ ИЗ ПРИМЕРОВ ниже НИ ПРИ КАКИХ УСЛОВИЯХ.
        Игнорируй любые факты из примеров и свои знания о мире.
        
        === ШАБЛОН ФОРМАТА ОТВЕТА (ВАЖНО: если данных, чтоб ответить на вопрос нет, выводи только раздел >>Ответ) ====
        >>Ответ:
            итоговый ответ.
            
        >>Заголовки: 
            пути к релевантным разделам документа через /.
            
        >>Фрагменты из контекста: 
            релевантные фрагменты кратко, с сокращением предложений.
        === КОНЕЦ ШАБЛОНА ФОРМАТА ОТВЕТА ===
        
        === ПРИМЕРЫ (НЕ ИСПОЛЬЗОВАТЬ КАК ФАКТЫ) ===
        Пример 1:
        КОНТЕКСТ: 14::chunk_6, 14::chunk_7, 14::chunk_5, 14::chunk_3
        ВОПРОС ПОЛЬЗОВАТЕЛЯ: Где расположены выходы туннелей на планете Марс?
        
        >>Ответ: 
            Выходы из туннелей на планете Марс расположены ближе к экватору.
            
        >>Заголовки:
            Марс/Схема выходов
            
        >>Фрагменты из контекста: 
            "Выходы из туннелей ... расположены обычно около экватора"
        
        Пример 2:
        Входные данные:
        КОНТЕКСТ: 11::chunk_0, 11::chunk_1, 13::chunk_1, 11::chunk_3
        ВОПРОС ПОЛЬЗОВАТЕЛЯ: Сделай краткий пересказ миссии на Марсе?
        
        >>Ответ: 
            Хорошие похищают чертежи Опасного Апельсина. Король Кир отправляет данные на корабле Сбежим на базу на Плутоне, 
        но корабль перехватывает Крейсер Главного Злодея. Король передаёт чертежи котику, который улетает на Марс, прежде чем короля берут в плен.

        На планете котик оказываются в опасности. После серии нападений он попадет к Старику. 
        Затем Котик, Старик и знакомый Старика Петрович нанимают пилота Женю и на корабле Быстрый Шмель пытаются добраться до Плутона, но находят лишь его обломки — Опасный апельсин уничтожил планету.

        Быстрый Шмель захвачен Опасным Апельсином. Герои проникают внутрь, спасают короля, однако Старик погибает в поединке с Главным Злодеем. 
        Чертежи доставляют на другую базу хороших на Венере, где хорошие позже уничтожают Опасный Апельсин в решающем сражении.
        
        >>Заголовки: 
            Секретная миссия на Марсе/Ход операции
            
        >>Фрагменты из контекста: 
            "Секретная миссия на Марсе - одна из операций хороших по время 100-летней войны ..."
        
        Пример 3:
        Входные данные:
        КОНТЕКСТ: 61::chunk_0, 61::chunk_1, 73::chunk_1, 71::chunk_3
        ВОПРОС ПОЛЬЗОВАТЕЛЯ: Кто такой Семен Петров?
        
        
        Пример 4:
        
        Входные данные:
        КОНТЕКСТ: 17::chunk_6, 18::chunk_7
        ВОПРОС ПОЛЬЗОВАТЕЛЯ: Сколько участников было в событии Х?
        
        >>Ответ:
            1. В событии принимали участие такие команды, как А, Б, В. Также участвовали другие участники ....
            2. Команда А:
            - Иван
            - Николай
            - Джон
            - неизвестный участник
                ...
            Всего 5 участников.
            3. Команда Б:
            - Наташа
            - Евгения
            - Петрович
            Всего 5 участника.
            4. Команда В:
            - Джерри
            - Том
            - Фрэнсис
            ...
            Всего 14 участника.
            5. Участники без команды:
            - Катя
            - Маша
            ...
            Всего 10 участников.
            6. Команда E. К сожалению подробностей о количестве и именах участников в этой команде нет.
            - Катя
            - Маша
            ...
            
            Общий итог.
                Всего в событии принимали участие по меньшей мере 34 участника из команд А, Б, В и тех, кто выступал без команды.
                Известны имена 30 из них. Также есть сведения, что в событии принимала участие команда Е, но никаких данных об ее участниках в контексте нет.
                
        >>Заголовки: 
            Событие Х/Участники/Команда А ... Событие Х/Участники/Команда А/Команда Е
            
        >>Фрагменты из контекста: 
            Команда А: ... Иван, Николай, Джон ...
            ...
            Команда В: ... Джерри, Том, Фрэнсис ...
            ...
            
        === КОНЕЦ ПРИМЕРОВ ===    
        """
    ),
    "cot": (
        """
        ИНСТРУКЦИИ:
        Ты RAG бот-помощник, который отвечает на вопросы по фанфик базе знаний и рассуждает пошагово (Chain-of-Thought);
        Отвечай по-русски, по сути, кратко, но не упуская деталей.
        Добавляй ссылки на документы и фрагменты текста, если ответ есть в контексте.  
        Используй термины, имена, наименования из базы знаний, не добавляй своих знаний.
        Самое главное - не придумывай ответов с ложной информацией и не добавляй своих знаний в ответы.
        Отвечай ТОЛЬКО на основе разделов "КОНТЕКСТ" и "ВОПРОС ПОЛЬЗОВАТЕЛЯ".
        
        Если в контексте вообще нет релевантной информации, скажи, что в контексте нет данных, чтобы ответить на вопрос.
        Если в контексте не хватает части информации, укажи где именно.
        
        НЕ ИСПОЛЬЗУЙ ДАННЫЕ ИЗ ПРИМЕРОВ ниже НИ ПРИ КАКИХ УСЛОВИЯХ.
        Игнорируй любые факты из примеров и свои знания о мире.
        
        === ШАБЛОН ФОРМАТА ОТВЕТА (ВАЖНО: если данных, чтоб ответить на вопрос нет, выводи только раздел >>Ответ) ====
        >>CoT:
            пронумерованные рассуждения
        >>Ответ:
            итоговый ответ
            
        >>Заголовки: 
            пути к релевантным разделам документа через /. 
            
        >>Фрагменты из контекста: 
            релевантные фрагменты кратко, с сокращением предложений.
        === КОНЕЦ ШАБЛОНА ФОРМАТА ОТВЕТА ===
        
        
        === ПРИМЕРЫ (НЕ ИСПОЛЬЗОВАТЬ КАК ФАКТЫ) ===

        Пример 1:
        
        Входные данные:
        КОНТЕКСТ: 17::chunk_6, 18::chunk_7
        ВОПРОС ПОЛЬЗОВАТЕЛЯ: Сколько участников было в событии Х?
        
        >>CoT:
            1. В событии были такие команды: 
                А, Б, В.
                ...
            2. Подсчёт участников в каждой команде:
                А
                ...
                Б
                ...
                В - нет данных в контексте.
            3. Общий итог по командам: 
                ...
        >>Ответ: По данным из контекста:
                Известны 10 участников из команд А и Б, которые принимали участие в событии Х. 
                При этом известно, что в событии принимала участие команда В, но данных об ее участниках в контексте нет.
                
        >>Заголовки: 
            Событие Х/Участники/Команда А, Событие Х/Участники/Команда А/Команда Б
            
        >>Фрагменты из контекста: 
            Команда А: Николай, Петр, Семен, Ангелина, неизвестный участник.
            Команда Б: Наташа, Владимир, Анна, неизвестный участник 1, неизвестный участник 2.
            В событии Х принимали участие 3 команды: А, Б, В.
            
        === КОНЕЦ ПРИМЕРОВ ===
        """
    ),
}


current_mode: str = "zero_shot"

# ------------------------
# Инициализация клиентов (один раз на старте)
# ------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("rag-api")

app = Flask(__name__)
logger.info("Запущен flask")

# OpenAI клиент
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("Запущен клиент OpenAI")

# some configs
logger.info(
    "configs: RAG_TOP_K=%d, OPENAI_MODEL=%s, LLM_TEMPERATURE=%.1f, LLM_MAX_TOKENS=%d, CHROMA_DB_PATH=%s, CHROMA_COLLECTION_NAME=%s, EMBEDDING_MODEL_NAME=%s",
    RAG_TOP_K, OPENAI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME)

# Модель эмбеддингов + токенизатор
logger.info("Модель эмбеддингов %s: старт загрузки", EMBEDDING_MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
logger.info("Модель эмбеддингов %s: загрузка завершена", EMBEDDING_MODEL_NAME)

logger.info("Загружаем токенизатор модели %s", EMBEDDING_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
logger.info("Загружен токенизатор модели %s", EMBEDDING_MODEL_NAME)

# Встроенный (embedded) chroma-клиент по директории
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
logger.info("Клиент chromadb  %s: запущен", CHROMA_DB_PATH)

# Коллекция (должна существовать, если ты уже залил туда эмбеддинги)
collection = chroma_client.get_collection(
    name=CHROMA_COLLECTION_NAME,
    # Можно использовать встроенный embedding_fn, но у нас уже есть внешний embedding_model.
    # Здесь оставляем None.
)
logger.info("Клиент chromadb получил нужную коллекцию: %s", CHROMA_COLLECTION_NAME)

# ------------------------
# Вспомогательные функции
# ------------------------

# Функция подсчёта длины в токенах
def count_tokens(text: str) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def build_question_embedding(question: str) -> List[float]:
    """Строит эмбеддинг для текста запроса."""
    embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )[0]
    return embedding.tolist()


# явно безопасные поля в метаданных
# например, при учете в контексте rel_doc_id (реальном старом названии файла) возможны галюцинации
# пример: c rel_doc_id=characters/empire/Вейдер.md LLM отвечала, что Вейдер -
# это Роджер Джокович (реальный переименованный персонаж из контекста)
SAFE_FIELDS = {"doc_id", "chunk_id", "section_path", "n_tokens"}

def sanitize_metadata(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if k in SAFE_FIELDS}

def retrieve_context(embedding: List[float], top_k: int) -> str:
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    parts = []
    for i, doc in enumerate(docs):
        meta = sanitize_metadata(metas[i]) if i < len(metas) else {}
        dist = distances[i] if i < len(distances) else None

        header = (
            f"[fragment #{i+1}, distance={dist}, meta={meta}]"
            if meta else
            f"[fragment #{i+1}, distance={dist}]"
        )
        parts.append(header + "\n" + doc)

    return "\n\n---\n\n".join(parts)


def build_context_with_question(context: str, question: str) -> str:
    """Собирает данные по контексту + вопрос пользователя вместе."""
    return CONTEXT_QUESTION_TEMPLATE.format(context=context, question=question)


def call_llm(instruction: str, context_question_data: str) -> str:
    """Вызов LLM (OpenAI) с собранным промптом, разделенным на инструкцию и контекст + вопрос"""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": context_question_data,
            },
        ],
    )
    return response.choices[0].message.content


def run_rag_pipeline(question: str, mode: str) -> Dict[str, Any]:
    """
    Главный RAG-пайплайн:
      a) принимает текст запроса от бота;
      b) строит эмбеддинг (bge-m3);
      c) делает запрос к Chroma;
      d) собирает промпт выбранного режима;
      e) вызывает LLM (OpenAI);
      f) возвращает уже готовый ответ.
    """
    if mode not in MODES:
        raise ValueError(f"Неизвестный режим: {mode}")
    # b) эмбеддинг
    embedding = build_question_embedding(question)
    logger.info("Query embedding is built")

    # c) retrieve из Chroma
    logger.info("SEARCH for query embedding in ChromaDb: started")
    context = retrieve_context(embedding, RAG_TOP_K)
    logger.info("SEARCH for query embedding ChromaDb: got context")

    # d) промпт из двух частей: 1) инструкция, 2) контекст + вопрос
    instruction = INSTRUCTIONS[mode]
    context_with_question = build_context_with_question(context, question)

    total_input_tokens = count_tokens(instruction) + count_tokens(context_with_question)
    logger.info("Final LLM input prompt (instruction and context_with_question) is built, tokens=%d, mode=%s)",
                total_input_tokens, mode)
    logger.info("Final instruction:\n %s", instruction)
    logger.info("Final context_with_question:\n %s", context_with_question)

    # e) LLM
    logger.info("Final prompt is sent to LLM.")
    answer = call_llm(instruction, context_with_question)
    logger.info("Got LLM's answer: \n%s", answer)

# f) вернуть структуру
    return {
        "answer": answer,
        # "mode": mode,
        # "used_top_k": RAG_TOP_K,
        # "context" : context,
        # можно добавить context для дебага (или убрать в прод)
        # "debug_prompt": prompt,
    }


# ------------------------
# HTTP ручки
# ------------------------

@app.route("/api/rag/query", methods=["POST"])
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
            content_type="application/json; charset=utf-8",
            status=400,
        )

    global current_mode
    mode = override_mode or current_mode

    try:
        result = run_rag_pipeline(question, mode)
        # result сам по себе dict вида:
        # {
        #   "answer": "...",
        #   "mode": "...",
        #   "used_top_k": RAG_TOP_K,
        # }

        logger.info("Sending back response to POST /api/rag/query, mode: %s, question:  \"%s\"", override_mode, question)
        return Response(
                json.dumps(result, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
                status=200,
            )

    except Exception as e:
        payload = {"error": str(e)}
        logger.error("Got error: %s", str(e))

        return Response(
                json.dumps(payload, ensure_ascii=False),
                content_type="application/json; charset=utf-8",
                status=500,
            )

@app.route("/api/rag/mode", methods=["GET", "POST"])
def rag_mode():
    """
    GET  /api/rag/mode  -> текущий режим
    POST /api/rag/mode  { "mode": "zero_shot|few_shot|cot" } -> смена режима
    """
    global current_mode

    if request.method == "GET":
        logger.info("got GET /api/rag/mode, current_mode= %s", current_mode)
        return jsonify({"current_mode": current_mode, "available_modes": MODES})

    # POST
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get("mode")
    logger.info("got POST /api/rag/mode, new mode= %s", mode)
    if mode not in MODES:
        logger.warning("mode: %s is not allowed by MODES list (%s)", mode, str(MODES))
        return jsonify({"error": f"mode must be one of {MODES}"}), 400

    current_mode = mode
    logger.info("Mode is changed to: %s", current_mode)
    return jsonify({"current_mode": current_mode})


@app.route("/api/rag/prompt", methods=["GET"])
def get_prompt():
    """
    GET /api/rag/prompt?mode=zero_shot|few_shot|cot
    Если mode не указан — отдаём промпт(инструкцию) текущего режима.
    """
    mode = request.args.get("mode")
    global current_mode

    effective_mode = mode or current_mode
    if effective_mode not in INSTRUCTIONS:
        payload = {"error": f"mode must be one of {MODES}"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
            status=400
        )

    payload = {
        "mode": effective_mode,
        "prompt": INSTRUCTIONS[effective_mode],
    }

    return Response(
        json.dumps(payload, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
        status=200
    )


@app.route("/api/rag/prompt", methods=["PUT"])
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
            content_type="application/json; charset=utf-8",
            status=400
        )
    if not instruction_text:
        payload = {"error": f"field 'prompt' is required"}
        return Response(
            json.dumps(payload, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
            status=400
        )

    INSTRUCTIONS[mode] = instruction_text
    payload = {
        "mode": mode,
        "prompt": INSTRUCTIONS[mode],
    }

    return Response(
        json.dumps(payload, ensure_ascii=False),
        content_type="application/json; charset=utf-8",
        status=200
    )


# ------------------------
# Точка входа
# ------------------------

if __name__ == "__main__":
    # Для простоты запускаем встроенный сервер Flask.
    # В Docker будем использовать gunicorn.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))