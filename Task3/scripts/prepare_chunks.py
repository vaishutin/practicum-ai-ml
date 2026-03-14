#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import time
from typing import List, Tuple
from transformers import AutoTokenizer
try:
    from langchain.text_splitter import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )
except ModuleNotFoundError:  # langchain>=0.2 split
    from langchain_text_splitters import (  # type: ignore
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

HEADER_TOKENS_DEFAULT = 20
MIN_SECTION_TOKENS = 350
MAX_OVERLAP_CHARS = 1000


def normalize_text(text: str) -> str:
    """Нормализация переносов строк и пробелов."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Сжимаем тройные и более переносы до двойных
    text = "\n\n".join(block.strip() for block in text.split("\n\n") if block.strip())
    return text.strip()


def count_tokens(text: str, tokenizer) -> int:
    """Подсчёт длины текста в токенах."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разбор .md: #/##/### и чанкинг по токенам в JSONL"
    )
    parser.add_argument(
        "-d", "--docs-path", type=Path, default="../task2/knowledge_base",
        help="Каталог с .md файлами (рекурсивно)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default="./prepared",
        help="Каталог, куда положить выходной JSONL-файл"
    )
    parser.add_argument(
        "--model-name", type=str,
        default="BAAI/bge-m3",
        help="Модель для токенизатора HuggingFace"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Максимум токенов в чанке"
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=64,
        help="Перекрытие чанков в токенах"
    )
    return parser.parse_args()


def compute_overlap_tokens(
    prev_text: str,
    curr_text: str,
    tokenizer,
    max_overlap_chars: int = MAX_OVERLAP_CHARS,
) -> int:
    """
    Ищем самый длинный суффикс prev_text, который совпадает с префиксом curr_text.
    Возвращаем длину этого пересечения в токенах.
    """
    if not prev_text or not curr_text:
        return 0

    max_len = min(len(prev_text), len(curr_text), max_overlap_chars)

    overlap_text = ""
    # идём от большого к маленькому, первый матч — самый длинный
    for k in range(max_len, 0, -1):
        if prev_text[-k:] == curr_text[:k]:
            overlap_text = prev_text[-k:]
            break

    if not overlap_text:
        return 0

    return count_tokens(overlap_text, tokenizer)


def split_section_into_records(
    section_text: str,
    section_path: list,
    doc_id: int,
    rel_doc_id: str,
    text_splitter,
    chunk_idx_start: int,
    tokenizer,
    section_tokens: int,
    max_tokens: int,
) -> Tuple[List[dict], int]:
    records = []

    # Берём первую строку как заголовок, остальной текст режем без неё
    header = ""
    if section_tokens > max_tokens:
        lines = section_text.splitlines()
        header = lines[0].strip()
        rest_text = "\n".join(lines[1:]).lstrip()
        chunks = text_splitter.split_text(rest_text)
    else:
        chunks = [section_text]

    prev_text = None
    prev_end_token = 0
    chunk_idx = chunk_idx_start
    overlap_tokens = 0

    for ch_text in chunks:
        # Сколько токенов в самом чанке
        if header != "":
            ch_with_header = header + "\n\n" + ch_text
            n_tokens = count_tokens(ch_with_header, tokenizer)
        else:
            n_tokens = section_tokens
            ch_with_header = ch_text

        if prev_text is None:
            # Первый чанк секции — без overlap
            start_token = 0
        else:
            # Считаем overlap по строкам между prev и current
            # Overlap применяется только если секция больше размера чанка
            overlap_tokens = compute_overlap_tokens(prev_text, ch_text, tokenizer)
            if overlap_tokens > 0:
                print(f"found overlap tokens: {overlap_tokens}")
            # Стартуем "назад" на overlap_tokens относительно конца прошлого чанка
            start_token = max(0, prev_end_token - overlap_tokens)

        end_token = start_token + n_tokens

        record = {
            "chunk_id": f"{doc_id}::chunk_{chunk_idx}",
            "metadata": {
                "doc_id": doc_id,
                "rel_doc_id": rel_doc_id,
                "section_path": section_path,
                "n_tokens": n_tokens,
                "overlap_tokens": overlap_tokens,
                "start_token": start_token,
                "end_token": end_token,
            },
            "text": ch_with_header,
        }

        records.append(record)
        chunk_idx += 1

        prev_text = ch_text
        prev_end_token = end_token

    return records, chunk_idx


def collect_md_files(docs_path: Path) -> List[Path]:
    md_files: List[Path] = []
    for root, _, files in os.walk(docs_path):
        for name in files:
            if name.lower().endswith(".md"):
                md_files.append(Path(root) / name)
    md_files.sort()
    return md_files


def main():
    t0 = time.perf_counter()

    args = parse_args()

    docs_path: Path = args.docs_path
    out_dir: Path = args.out_dir

    if not docs_path.exists():
        raise SystemExit(f"Каталог {docs_path} не существует")

    print(f"[INFO] Загружаем токенизатор {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # --- Проверка допустимого значения токенов в чанке для выбранной модели ---
    model_max = tokenizer.model_max_length
    if model_max and model_max != -1:  # -1 означает unlimited (редко бывает)
        if args.max_tokens > model_max:
            raise SystemExit(
                f"ОШИБКА: max-tokens={args.max_tokens} превышает лимит модели "
                f"{model_max} токенов (model: {args.model_name}).\n"
                f"Исправьте параметр: --max-tokens <= {model_max}"
            )
    # -----------------------
    # Сплиттер по заголовкам Markdown
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
    )

    # Рекурсивный сплиттер с учётом токенов и перекрытия
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.max_tokens - HEADER_TOKENS_DEFAULT,
        chunk_overlap=args.overlap_tokens,
        separators=["\n\n", "\n", " ", ""],
        length_function=lambda txt: count_tokens(txt, tokenizer),
    )

    # Формируем имя результирующего файла
    model_clean = args.model_name.replace("/", "--")
    outfile_name = f"{model_clean}_{args.max_tokens}_{args.overlap_tokens}.jsonl"
    out_path = out_dir / outfile_name

    # Создаём каталог
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Файл вывода: {out_path}")

    md_files = collect_md_files(docs_path)

    total_chunks = 0

    doc_id = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for md_file in md_files:
            doc_id += 1
            rel_doc_id = str(md_file.relative_to(docs_path))
            print(f"[INFO] Начинаем чанкование {rel_doc_id}")

            raw = md_file.read_text(encoding="utf-8")
            text = normalize_text(raw)

            # 1) Разбиваем документ по заголовкам (#, ##, ###)
            # header_splitter возвращает список "нод":
            # {"content": "...", "metadata": {"h1": "...", "h2": "...", ...}}
            nodes = header_splitter.split_text(text)
            if not nodes:
                continue

            chunk_idx = 0

            combined_section_text = ""
            combined_section_tokens = 0
            combined_section_paths = []
            for node in nodes:
                section_text: str = node.page_content
                md_meta: dict = node.metadata or {}

                # Собираем section_path [], [H1], [H1, H2], [H1, H2, H3]
                section_path = []
                section_text_with_headers = ""
                for key in ("h1", "h2", "h3"):
                    if key in md_meta and md_meta[key]:
                        section_path.append(md_meta[key])
                        section_text_with_headers += md_meta[key] + ". "

                section_text_with_headers = section_text_with_headers[:-2]
                section_text_with_headers += "\n\n" + section_text

                section_tokens = count_tokens(section_text_with_headers, tokenizer)
                if combined_section_tokens + section_tokens < MIN_SECTION_TOKENS:
                    combined_section_paths.append(section_path)
                    combined_section_text += section_text_with_headers + "\n\n"
                    combined_section_tokens += section_tokens
                else:
                    # 2) Режем пред.комбинированную секцию на чанки сплиттером
                    if combined_section_tokens > 0:
                        records, chunk_idx = split_section_into_records(
                            section_text=combined_section_text,
                            section_path=combined_section_paths,
                            doc_id=doc_id,
                            rel_doc_id=rel_doc_id,
                            text_splitter=text_splitter,
                            chunk_idx_start=chunk_idx,
                            tokenizer=tokenizer,
                            section_tokens=combined_section_tokens,
                            max_tokens=args.max_tokens,
                        )

                        for record in records:
                            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            total_chunks += 1

                    combined_section_paths = [section_path]
                    combined_section_text = section_text_with_headers + "\n\n"
                    combined_section_tokens = section_tokens

            if combined_section_tokens > 0:
                records, chunk_idx = split_section_into_records(
                    section_text=combined_section_text,
                    section_path=combined_section_paths,
                    doc_id=doc_id,
                    rel_doc_id=rel_doc_id,
                    text_splitter=text_splitter,
                    chunk_idx_start=chunk_idx,
                    tokenizer=tokenizer,
                    section_tokens=combined_section_tokens,
                    max_tokens=args.max_tokens,
                )

                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print(f"[DONE] Файлов: {len(md_files)}, чанков: {total_chunks}")
    print(f"[OUT] {out_path}")
    print(f"Время потрачено: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
