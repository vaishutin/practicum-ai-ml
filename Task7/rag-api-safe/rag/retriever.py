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
    def __init__(self, chroma_client: Any, chroma_collection_name: str, embedder: Any, top_k: int = 6):
        self._chroma_client = chroma_client
        self.chroma_collection_name = chroma_collection_name
        self._embedder = embedder
        self._top_k = top_k


    # ------------------------------------------------------------------
    # Публичный bulk-retrieval
    # ------------------------------------------------------------------
    def retrieve_topk_bulk(
            self,
            *,
            pipeline,
            questions: list[dict],
            k: int,
            include_text: bool = False,
    ) -> dict:
        """
        Возвращает:
        {
          "k": int,
          "results": [{"id": question_id,"results": [ ...items... ]} ... ]
        }
        """

        if not (1 <= k <= 50):
            raise ValueError("k must be in [1..50]")
        if not questions:
            raise ValueError("questions must be non-empty")

        qids: list[str] = []
        texts: list[str] = []
        for q in questions:
            if "id" not in q or "question" not in q:
                raise ValueError("each question must have 'id' and 'question'")
            qids.append(q["id"])
            texts.append(q["question"])

        # 1) считаем эмбеддинги для ВСЕХ вопросов разом (с нормализацией)
        embs = self._embedder.encode(
            texts,
            normalize_embeddings=True,
        )
        logger.info("retrieve_topk_bulk - embeddings are built")
        # SentenceTransformer вернёт numpy array (N x dim), Chroma ждёт list[list[float]]
        embs = embs.tolist()

        # 2) bulk query по эмбеддингам
        n_results = k + 3 # запас из-за SAFE filter
        chroma_collection = self._chroma_client.get_collection(name=self.chroma_collection_name,)
        logger.info(f"Sending topk={n_results} bulk query to ChromaDb")

        chroma_res = chroma_collection.query(
            query_embeddings=embs,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        logger.info("retrieve_topk_bulk - got results from ChromaDb")
        all_embedding_ids = chroma_res.get("ids", [])
        all_docs = chroma_res.get("documents", [])
        all_metas = chroma_res.get("metadatas", [])
        all_dist = chroma_res.get("distances", [])

        out = []
        logger.info(f"retrieve_topk_bulk - start post-processing results with top{k} chunks filter")

        for qi, qid in enumerate(qids):
            logger.info(f"Starting post-processing question {qi+1}/{len(qids)}: {qid}")
            items = self._postprocess_one(
                qid=qid,
                embedding_ids=all_embedding_ids[qi] if qi < len(all_embedding_ids) else [],
                docs=all_docs[qi] if qi < len(all_docs) else [],
                metas=all_metas[qi] if qi < len(all_metas) else [],
                distances=all_dist[qi] if qi < len(all_dist) else [],
                k=k,
                pipeline=pipeline,
                include_text=include_text,
            )
            out.append({"id": qid, "include_text": include_text, "results": items})

        return {"k": k, "results": out}


    # ------------------------------------------------------------------
    # Внутренняя обработка одного retrieval-результата (одного вопроса)
    # ------------------------------------------------------------------
    def _postprocess_one(
            self,
            qid: str,
            embedding_ids: list[str],
            docs: list[str],
            metas: list[dict],
            distances: list[float],
            *,
            k: int,
            pipeline,
            include_text: bool = False,
    ) -> list[dict]:
        """
        Возвращает list[dict] для одного вопроса:
          {
            "embedding_id": str,
            "rank": int,
            "distance": float | None,
            "meta": {SAFE_FIELDS},
            "text": str
          }
        """
        items: list[dict] = []
        rank = 0
        for i, doc in enumerate(docs):
            embedding_id = embedding_ids[i]
            meta = self.sanitize_metadata(metas[i] if i < len(metas) else {}, i, qid)
            chunk_id = meta.get("chunk_id")
            dist = distances[i] if i < len(distances) else None

            if getattr(pipeline, "retriever_chunks_filter", False) and self.chunk_is_not_safe(doc, chunk_id):
                continue

            if getattr(pipeline, "retriever_text_cleaner", False):
                doc = self.clean_bad_words(doc, chunk_id)

            rank += 1
            item = {
                "embedding_id": embedding_id,
                "rank": rank,
                "distance": dist,
                "meta": meta,
            }
            if include_text:
                item["text"] = doc
            items.append(item)

            if rank >= k:
                break
        logger.info(f"Question postprocessing finished: {len(items)} chunks returned")
        return items

    def retrieve_context(self, question: str) -> str:
        # b) эмбеддинг
        embedding = self.build_question_embedding(question)
        logger.info("Query embedding is built")

        # ====================Отладочный код перед get_collection
        all_collections = self._chroma_client.list_collections()
        logger.info(f"Доступные коллекции: {[c.name for c in all_collections]}")

        # c) retrieve from Chroma
        logger.info(f"SEARCH for query embedding in chroma_collection={self.chroma_collection_name} started")
        # Коллекция (должна существовать, предполагаем, что уже существует)
        chroma_collection = self._chroma_client.get_collection(
            name=self.chroma_collection_name,
            # Можно использовать встроенный embedding_fn, но у нас уже есть внешний embedding_model.
            # Здесь оставляем None.
        )

        results = chroma_collection.query(
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
            meta = self.sanitize_metadata(metas[i], i) if i < len(metas) else {}
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
    def sanitize_metadata(meta: dict, chunk_num: int, qid:str ="0") -> dict:
        try:
            return {k: v for k, v in meta.items() if k in SAFE_FIELDS}
        except Exception as e:
            logger.error(f"sanitize_metadata: {qid}, chunk={chunk_num}, {e}")
            return {}

    def build_question_embedding(self, question: str) -> List[float]:
        """Строит эмбеддинг для текста запроса."""
        embedding = self._embedder.encode(
            [question],
            normalize_embeddings=True,
        )[0]
        return embedding.tolist()