#!/usr/bin/env python3
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self,
                 embedder: Any,
                 chroma_client: Any,
                 chunks_path: Path,
                 chroma_dir: Path,
                 chroma_collection_name: str = "rag_bot_kb",
                 embedder_model: str = "BAAI/bge-m3",
                 max_tokens: int = 256,
                 overlap_tokens: int = 64,
                 ):
        self._embedder = embedder
        self._chroma_client = chroma_client
        self.chunks_path = chunks_path
        self.chroma_dir = chroma_dir
        self.chroma_collection_name = chroma_collection_name
        self.embedder_model = embedder_model
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def load_chunks(self):
        """
        Ожидаемый формат строки:
          {"chunk_id": "...",  "metadata": {...}, "text": "..."}
        """
        chunk_ids = []
        texts = []
        metadata = []

        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                # Нужен консистентный id, который будет первичным ключем для embeddings.embedding_id
                # на основании пути документа + чанка, т.е. "events/Битва_при_Джакку.md::chunk_0"
                # При этом в retriever мы не будем использовать имя документа, а ограничимся metadata.chunk_id вида 11::chunk_0
                chunk_id = str(obj["embedding_id"])
                text = obj["text"]
                data_token = obj.get("metadata", {})

                if not isinstance(data_token, dict):
                    data_token = {"metadata": data_token}

                data_token = self.normalize_metadata(data_token)

                chunk_ids.append(chunk_id)
                texts.append(text)
                metadata.append(data_token)

        return chunk_ids, texts, metadata

    @staticmethod
    def filter_chunks_by_rel_doc_ids(
            chunk_ids: list[str],
            texts: list[str],
            metadata: list[dict],
            source_paths: set[str],
    ):
        """
        Фильтрует загруженные чанки по metadata["rel_doc_id"].
        Возвращает те же структуры: (chunk_ids, texts, metadata)
        """
        if not source_paths:
            return [], [], []

        f_chunk_ids = []
        f_texts = []
        f_metadata = []

        for cid, text, meta in zip(chunk_ids, texts, metadata):
            source_path = meta.get("rel_doc_id")
            if source_path not in source_paths:
                continue

            f_chunk_ids.append(cid)
            f_texts.append(text)
            f_metadata.append(meta)

        return f_chunk_ids, f_texts, f_metadata

    @staticmethod
    def normalize_metadata(md: dict) -> dict:
        """Превращаем списки/сложные типы в строки, чтобы Chroma их съела."""
        norm = {}
        for k, v in md.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                norm[k] = v
            elif isinstance(v, list):
                # если список строк — склеиваем
                if all(isinstance(x, str) for x in v):
                    norm[k] = ", ".join(v)
                else:
                    norm[k] = str(v)
            else:
                # на всякий случай — всё остальное в строку
                norm[k] = str(v)
        return norm


    # -----------------------------
    # Delete
    # -----------------------------
    def delete_by_source_paths(self, source_paths: Iterable[str]) -> int:
        """
        Удаляем все чанки для указанных source_path.

        Реализация сделана максимально совместимой:
        1) пытаемся delete(where={"source_path": {"$in": [...]}})
        2) если не поддерживается — делаем по одному path:
           get(where={"source_path": path}) -> ids -> delete(ids=...)
        Возвращаем "сколько ids мы попытались удалить" (приближенно).
        """
        paths = [p for p in source_paths if p]
        if not paths:
            return 0

        # Попытка bulk delete по $in
        chroma_collection = self._chroma_client.get_collection(name=self.chroma_collection_name)
        try:
            chroma_collection.delete(where={"rel_doc_id": {"$in": paths}})
            logger.info(f"Bulk delete(where $in {paths}) succeeded")
            # Chroma обычно не возвращает count. Вернём len(paths) как approx.
            return len(paths)
        except Exception as e:
            logger.error("Bulk delete(where $in) not supported, fallback to per-path. err=%s", e)

        deleted_ids = 0
        for p in paths:
            ids = self._get_ids_by_source_path(p, chroma_collection=chroma_collection)
            if ids:
                chroma_collection.delete(ids=ids)
                logger.info(f"delete(ids={ids}) succeeded")
                deleted_ids += len(ids)
        return deleted_ids


    @staticmethod
    def _get_ids_by_source_path(source_path: str, chroma_collection: Any) -> list[str]:
        # Пробуем стандартный get(where=...)
        try:
            res = chroma_collection.get(where={"rel_doc_id": source_path}, include=["ids"])
            logger.info(f"collection.get(where=rel_doc_id = {source_path}) succeeded: {res}", )
            ids = res.get("ids") or []
            # В некоторых версиях ids может быть list[str] или list[list[str]] — нормализуем
            if ids and isinstance(ids[0], list):
                # type: ignore[assignment]
                ids = [x for sub in ids for x in sub]
            return ids
        except Exception as e:
            logger.error("collection.get(where=...) failed for %s err=%s", source_path, e)
            return []

    # -----------------------------
    # Misc
    # -----------------------------
    def index_size_bytes(self) -> int:
        total = 0
        if not self.chroma_dir.exists():
            logger.warning(f"Directory {self.chroma_dir} not found, assuming empty.")
            return 0
        for p in self.chroma_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total


    def build_index(self, source_paths: set[str]) -> int:
        t0 = time.perf_counter()

        if not self.chunks_path.exists():
            raise SystemExit(f"Файл с чанками не найден: {self.chunks_path}")

        logger.info(f"Файл чанков:      {self.chunks_path}")
        logger.info(f"embedder_model:   {self.embedder_model}")
        logger.info(f"max_tokens:       {self.max_tokens}")
        logger.info(f"overlap:          {self.overlap_tokens}")
        logger.info(f"Collection name:  {self.chroma_collection_name}")

        # 1. Грузим все чанки
        logger.info("Читаем чанки ...")
        chunk_ids, texts, metadatas = self.load_chunks()
        # 2. Фильтрация по файлам
        chunk_ids, texts, metadatas = self.filter_chunks_by_rel_doc_ids(chunk_ids, texts, metadatas, source_paths)
        total = len(chunk_ids)
        logger.info(f"Найдено новых/измененных чанков для обновления в индекс: {total}")
        if total == 0:
            logger.warning("Нет новых/измененных для обновления индекса, выходим.")
            return 0

        logger.info("Считаем эмбеддинги ...")
        embeddings = self._embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Создаем коллекцию для первого запуска
        chroma_col = self._chroma_client.get_or_create_collection(
            name=self.chroma_collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedder_model,
                "max_chunk_tokens": self.max_tokens,
                "max_chunk_overlap_tokens": self.overlap_tokens,
            },
        )

        logger.info("Записываем документы в коллекцию ...")

        # делаем upsert, если это возможно
        if hasattr(chroma_col, "upsert"):
            chroma_col.upsert(
                ids=chunk_ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
        else:
            # fallback, если upsert нет (менее удобно)
            chroma_col.add(
                ids=chunk_ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )


        logger.info("Готово.")
        logger.info(f"Время потрачено: {time.perf_counter() - t0:.2f}s")

        return total
