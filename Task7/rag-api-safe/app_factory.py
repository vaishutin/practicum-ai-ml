import os
from typing import Any

from flask import Flask

import chromadb
from chromadb.config import Settings
from api.routes import api_bp
from api.routes_bulk import api_bulk_bp
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import ChromaRetriever
from rag.pipeline import RagPipeline
from rag.bulk_openai_client import BulkOpenAIClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-app-factory")

# Сначала загружаем публичный конфиг, потом секреты.
load_dotenv(".env")
load_dotenv(".env.secrets")

# ------------------------
# Конфиг
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "BAAI--bge-m3_1024_130")

CHROMA_MODE = os.getenv("CHROMA_MODE", "http")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chroma")
CHROMA_PORT = os.getenv("CHROMA_PORT", 8000)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))

# some configs
logger.info(
    "configs: RAG_TOP_K=%d, CHROMA_DB_PATH=%s, CHROMA_COLLECTION_NAME=%s, EMBEDDING_MODEL_NAME=%s",
    RAG_TOP_K, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL_NAME)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY не задан. См. README, как его получить и установить."
    )

class RagApp(Flask):
    openai_client: OpenAI
    bulk_openai_client: BulkOpenAIClient
    embedder: SentenceTransformer
    tokenizer: AutoTokenizer
    chroma_client: Any
    chroma_collection_name: str
    retriever: ChromaRetriever
    pipeline: RagPipeline

# ------------------------
# Инициализация клиентов (один раз на старте)
def create_app():
    app = RagApp(__name__)
    app.register_blueprint(api_bp)
    app.register_blueprint(api_bulk_bp)

    logger.info("Запущен flask")

    # OpenAI клиент
    app.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Запущен клиент OpenAI")

    # Bulk OpenAI клиент
    app.bulk_openai_client = BulkOpenAIClient(app.openai_client)
    logger.info("Инициализирована bulk обертка клиента OpenAI")

    # Эмбеддер + токенизатор
    logger.info("Модель эмбеддингов %s: старт загрузки", EMBEDDING_MODEL_NAME)
    app.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Модель эмбеддингов %s: загрузка завершена", EMBEDDING_MODEL_NAME)

    logger.info("Загружаем токенизатор модели %s", EMBEDDING_MODEL_NAME)
    app.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    logger.info("Загружен токенизатор модели %s", EMBEDDING_MODEL_NAME)

    logger.info(f"CHROMA_MODE = {CHROMA_MODE}, CHROMA_HOST = {CHROMA_HOST}, CHROMA_PORT = {CHROMA_PORT}")
    # Встроенный (embedded) chroma-клиент
    if CHROMA_MODE == "http":
        logger.info(f"Создается http-клиент Chroma для: {CHROMA_HOST}:{CHROMA_PORT}")
        app.chroma_client = chromadb.HttpClient(
                host=CHROMA_HOST,
                port=CHROMA_PORT,
                settings=Settings(anonymized_telemetry=False),
            )
    else:
        logger.info(f"Создается persistent-клиент Chroma для хранилища: {CHROMA_DB_PATH}")
        app.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    app.chroma_collection_name = CHROMA_COLLECTION_NAME
    app.retriever = ChromaRetriever(chroma_client=app.chroma_client,
                                    chroma_collection_name=app.chroma_collection_name,
                                    embedder=app.embedder, top_k=RAG_TOP_K)
    app.pipeline = RagPipeline(retriever=app.retriever,
                               tokenizer=app.tokenizer,
                               openai_client=app.openai_client,
                               current_mode="zero_shot")
    logger.info("Приложение загружено", )
    return app