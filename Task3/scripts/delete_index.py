#!/usr/bin/env python3
import argparse
from pathlib import Path

import chromadb


def main():
    parser = argparse.ArgumentParser(
        description="Удаляем коллекцию по имени с помощью Chroma API."
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=Path("./chroma_db"),
        help="Каталог для persistent-хранилища Chroma (по умолчанию ./chroma_db)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Имя коллекции вида BAAI--bge-m3_1024_132",
    )

    args = parser.parse_args()
    collection_name: str = args.collection_name
    print(f"[INFO] Collection name:  {collection_name}")

    # Chroma
    print(f"[INFO] Подключаемся к Chroma: {args.chroma_dir}")
    client = chromadb.PersistentClient(path=str(args.chroma_dir))

    print(f"[INFO] Удаляем коллекцию '{collection_name}'.")
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Старая коллекция '{collection_name}' удалена.")
    except Exception:
        print(f"[WARN] коллекции '{collection_name}' не найдено.")
        # Если её нет – это нормально
        pass

    print("[INFO] Готово.")


if __name__ == "__main__":
    main()
