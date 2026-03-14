import os

from flask import Flask

import chromadb
from api.routes import api_bp
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
from rag.retriever import ChromaRetriever
from rag.pipeline import RagPipeline


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
    embedder: SentenceTransformer
    tokenizer: AutoTokenizer
    chroma_client: chromadb.PersistentClient
    collection: chromadb.Collection
    retriever: ChromaRetriever
    pipeline: RagPipeline

# ------------------------
# Инициализация клиентов (один раз на старте)
def create_app():
    app = RagApp(__name__)
    app.register_blueprint(api_bp)
    logger.info("Запущен flask")

    # OpenAI клиент
    app.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("Запущен клиент OpenAI")

    # Эмбеддер + токенизатор
    logger.info("Модель эмбеддингов %s: старт загрузки", EMBEDDING_MODEL_NAME)
    app.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Модель эмбеддингов %s: загрузка завершена", EMBEDDING_MODEL_NAME)

    logger.info("Загружаем токенизатор модели %s", EMBEDDING_MODEL_NAME)
    app.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    logger.info("Загружен токенизатор модели %s", EMBEDDING_MODEL_NAME)

    # Встроенный (embedded) chroma-клиент
    app.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logger.info("Клиент chromadb  %s: запущен", CHROMA_DB_PATH)

    # Коллекция (должна существовать, предполагаем, что уже существует)
    app.collection = app.chroma_client.get_collection(
        name=CHROMA_COLLECTION_NAME,
        # Можно использовать встроенный embedding_fn, но у нас уже есть внешний embedding_model.
        # Здесь оставляем None.
    )
    logger.info("Клиент chromadb получил нужную коллекцию: %s", CHROMA_COLLECTION_NAME)

    app.retriever = ChromaRetriever(chroma_collection=app.collection, embedder=app.embedder, top_k=RAG_TOP_K)
    app.pipeline = RagPipeline(retriever=app.retriever,
                               tokenizer=app.tokenizer,
                               openai_client=app.openai_client,
                               current_mode="zero_shot")

    return app