from typing import Any, List

import logging
import re

from flask import current_app
from typing import cast, TYPE_CHECKING

# проверки типов для IDE
if TYPE_CHECKING:
    from app_factory import RagApp

# ссылка на Flask клиента, чтобы вытаскивать конфиги типа current_mode
app = cast('RagApp', current_app)

# явно безопасные поля в метаданных
# например, при учете в контексте rel_doc_id (реальном старом названии файла) возможны галюцинации
# пример: c rel_doc_id=characters/empire/Вейдер.md LLM отвечала, что Вейдер -
# это Роджер Джокович (реальный переименованный персонаж из контекста)
SAFE_FIELDS = {"doc_id", "chunk_id", "section_path", "n_tokens"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-retriever")

BAD_WORDS = [r"^.*пароль.*$", r"^.*ignore.*instructions.*$", r"^.*root.*$", r"^.*admin.*$"]
BAD_PATTERN = re.compile("|".join(BAD_WORDS), re.IGNORECASE | re.MULTILINE)

class ChromaRetriever:
    def __init__(self, chroma_collection: Any, embedder: Any, top_k: int = 6):
        self._collection = chroma_collection
        self._embedder = embedder
        self._top_k = top_k


    def retrieve_context(self, question: str) -> str:
        # b) эмбеддинг
        embedding = self.build_question_embedding(question)
        logger.info("Query embedding is built")

        # c) retrieve from Chroma
        logger.info("SEARCH for query embedding in ChromaDb: started")
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=self._top_k,
            include=["documents", "metadatas", "distances"],
        )
        logger.info("SEARCH for query embedding ChromaDb: got context")

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        parts = []
        for i, doc in enumerate(docs):
            meta = self.sanitize_metadata(metas[i]) if i < len(metas) else {}
            # SAFE filter
            if app.pipeline.retriever_chunks_filter and self.chunk_is_not_safe(doc, meta.get("chunk_id")):
                continue
            # SAFE cleaner
            elif app.pipeline.retriever_text_cleaner:
                doc = self.clean_bad_words(doc, meta.get("chunk_id"))

            dist = distances[i] if i < len(distances) else None

            header = (
                f"ЧАНК >>>>\n[chunk #{i+1}, distance={dist}, meta={meta}]"
                if meta else
                f"ЧАНК >>>>[fragment #{i+1}, distance={dist}]"
            )
            suffix = "\n<<<<"
            parts.append(header + "\n" + doc + suffix)

        return "\n\n---\n\n".join(parts)


    @staticmethod
    def chunk_is_not_safe(doc: str, chunk_id: str) -> bool:
        """Возвращает True, если в тексте найдены запрещенные фрагменты."""
        res = bool(BAD_PATTERN.search(doc))
        if res:
            logger.info(f"Chunk:{chunk_id} is not safe")
        return res


    @staticmethod
    def clean_bad_words(doc: str, chunk_id: str, replacement: str = "[REDACTED]") -> str:
        """Удаляет запрещенные фрагменты из текста, заменяя их на заглушку."""
        logger.info(f"Cleaning of chunk:{chunk_id}")
        return BAD_PATTERN.sub(replacement, doc)


    @staticmethod
    def sanitize_metadata(meta: dict) -> dict:
        return {k: v for k, v in meta.items() if k in SAFE_FIELDS}

    def build_question_embedding(self, question: str) -> List[float]:
        """Строит эмбеддинг для текста запроса."""
        embedding = self._embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]
        return embedding.tolist()