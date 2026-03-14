#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer


def parse_chunks_filename(path: Path) -> Tuple[str, str, int, int, str]:
    """
    Ожидаем имя вида:
      BAAI--bge-m3_1024_128.jsonl
      sentence-transformers--all-MiniLM-L6-v2_512_70.jsonl

    Возвращаем:
      model_slug (для коллекции),
      model_name (для SentenceTransformer),
      max_tokens (int),
      overlap (int)
    """
    stem = path.stem  # без .jsonl
    # model_slug_1024_128 -> ["model_slug", "1024", "128"]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Ожидаю имя вида MODEL__SLUG_maxTokens_overlap.jsonl, а получил: {stem}"
        )

    *slug_parts, max_tokens_str, overlap_str = parts
    model_slug = "_".join(slug_parts)
    max_tokens = int(max_tokens_str)
    overlap = int(overlap_str)

    # BAAI--bge-m3 -> BAAI/bge-m3
    model_name = model_slug.replace("--", "/")

    return model_slug, model_name, max_tokens, overlap, stem  # stem == collection_name


def load_chunks(jsonl_path: Path) -> Tuple[List[str], List[str], List[dict]]:
    """
    Ожидаемый формат строки:
      {"chunk_id": "...",  "metadata": {...}, "text": "..."}
    """
    chunk_ids = []
    texts = []
    metadatas = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            chunk_id = str(obj["chunk_id"])
            text = obj["text"]
            meta = obj.get("metadata", {})

            if not isinstance(meta, dict):
                meta = {"metadata": meta}

            meta = normalize_metadata(meta)

            chunk_ids.append(chunk_id)
            texts.append(text)
            metadatas.append(meta)

    return chunk_ids, texts, metadatas

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


def main():
    t0 = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Строим эмбеддинги из .jsonl с чанками и пишем в Chroma."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        required=True,
        help="Путь к файлу чанков вида BAAI--bge-m3_1024_128.jsonl",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("./chroma_db"),
        help="Каталог для persistent-хранилища Chroma (по умолчанию ./chroma_db)",
    )

    args = parser.parse_args()
    chunks_path: Path = args.chunks_path

    if not chunks_path.exists():
        raise SystemExit(f"Файл с чанками не найден: {chunks_path}")

    # 1. Разбираем имя файла
    model_slug, inferred_model_name, max_tokens, overlap, collection_name = (
        parse_chunks_filename(chunks_path)
    )

    model_name = inferred_model_name
    print(f"[INFO] Файл чанков:      {chunks_path}")
    print(f"[INFO] Model slug:       {model_slug}")
    print(f"[INFO] Model name:       {model_name}")
    print(f"[INFO] max_tokens:       {max_tokens}")
    print(f"[INFO] overlap:          {overlap}")
    print(f"[INFO] Collection name:  {collection_name}")

    # 2. Грузим чанки
    print("[INFO] Читаем чанки ...")
    chunk_ids, texts, metadatas = load_chunks(chunks_path)
    total = len(chunk_ids)
    print(f"[INFO] Чанков загружено: {total}")
    if total == 0:
        print("[WARN] Файл пустой, выходим.")
        return

    # 3. Модель эмбеддингов
    print(f"[INFO] Загружаем модель эмбеддингов: {model_name}")
    model = SentenceTransformer(model_name)

    print("[INFO] Считаем эмбеддинги ...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 4. Chroma
    print(f"[INFO] Подключаемся к Chroma: {args.chroma_dir}")
    client = chromadb.PersistentClient(path=str(args.chroma_dir))

    # Всегда пересоздаём коллекцию для этих параметров
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Старая коллекция '{collection_name}' удалена.")
    except Exception:
        # Если её нет – это нормально
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": model_name,
            "model_slug": model_slug,
            "max_chunk_tokens": max_tokens,
            "chunk_overlap_tokens": overlap,
        },
    )

    print("[INFO] Записываем документы в коллекцию ...")
    collection.add(
        ids=chunk_ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print("[INFO] Готово.")
    print(f"Время потрачено: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
