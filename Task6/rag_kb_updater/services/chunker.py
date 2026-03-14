#!/usr/bin/env python3
import json
import logging
import os
from pathlib import Path

import time
from typing import Any

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

Path("../knowledge_base"),
Path("./injections"),


KB_ROOT = Path(os.getenv("KB_ROOT", "../knowledge_base"))  # можно поменять
OUT_DIR = Path(os.getenv("OUT_DIR", "../prepared_chunks"))

class Chunker:
    def __init__(self,
                 tokenizer: Any,
                 embedder_model: str = "BAAI/bge-m3",
                 max_tokens: int = 256,
                 overlap_tokens: int = 64):
        self._tokenizer = tokenizer
        self.embedder_model = embedder_model
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @staticmethod
    def normalize_text(text: str) -> str:
        """Нормализация переносов строк и пробелов."""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Сжимаем тройные и более переносы до двойных
        text = "\n\n".join(block.strip() for block in text.split("\n\n") if block.strip())
        return text.strip()

    def compute_overlap_tokens(self, prev_text: str, curr_text: str, max_overlap_chars: int = 1000) -> int:
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

        return self.count_tokens(overlap_text)


    def split_section_into_records(
            self,
            section_text: str,
            section_path: list,
            doc_id: int,
            rel_doc_id: str,
            text_splitter,
            chunk_idx_start: int,
            section_tokens: int,
            max_tokens: int,
    ):
        records = []

        # Берём первую строку из текста, cплитим текст без заголовка
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
            # сколько токенов в самом чанке
            if header != "":
                ch_with_header = header + "\n\n" + ch_text
                n_tokens = self.count_tokens(ch_with_header)
            else:
                n_tokens = section_tokens
                ch_with_header = ch_text

            if prev_text is None:
                # первый чанк секции — без overlap
                start_token = 0
            else:
                # считаем overlap по строкам между prev и current
                # overlapping будет применяться, только если секция больше размера чанка
                overlap_tokens = self.compute_overlap_tokens(prev_text, ch_text)
                if overlap_tokens > 0:
                    logger.info(f"found overlap tokens: {overlap_tokens}")
                # стартуем "назад" на overlap_tokens относительно конца прошлого чанка
                start_token = max(0, prev_end_token - overlap_tokens)

            end_token = start_token + n_tokens

            record = {
                "chunk_id": f"{doc_id}::chunk_{chunk_idx}",
                "embedding_id": f"{rel_doc_id}::chunk_{chunk_idx}",
                "metadata": {
                    "chunk_id": f"{doc_id}::chunk_{chunk_idx}",
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


    def count_tokens(self, text: str) -> int:
        return len(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    @staticmethod
    def write_chunks_dummy(docs_paths: list[str]):
        return len(docs_paths)

    def prepare_chunks(self) -> int:
        t0 = time.perf_counter()

        docs_paths: list[Path]  = [KB_ROOT]
        out_dir: Path = OUT_DIR

        for docs_path in docs_paths:
            if not docs_path.exists():
                raise SystemExit(f"Каталог {docs_path} не существует")

        # --- Проверка допустимого значения токенов в чанке для выбранной модели ---
        model_max = self._tokenizer.model_max_length
        if model_max and model_max != -1:  # -1 означает unlimited (редко бывает)
            if self.max_tokens > model_max:
                raise SystemExit(
                    f"ОШИБКА: max-tokens={self.max_tokens} превышает лимит модели "
                    f"{model_max} токенов (model: {self.embedder_model}).\n"
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
            chunk_size=self.max_tokens - 20, # minus header tokens
            chunk_overlap=self.overlap_tokens,
            separators=["\n\n", "\n", " ", ""],
            length_function=lambda txt: self.count_tokens(txt),
        )

        # Формируем имя результирующего файла
        model_clean = self.embedder_model.replace("/", "--")
        outfile_name = f"{model_clean}_{self.max_tokens}_{self.overlap_tokens}.jsonl"
        out_path = out_dir / outfile_name

        # создаём каталог
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Файл вывода: {out_path}")

        md_files: list[Path] = []

        for docs_path in docs_paths:
            for root, _, files in os.walk(docs_path):
                for name in files:
                    if name.lower().endswith(".md"):
                        md_files.append(Path(root) / name)
        md_files.sort()

        total_chunks = 0

        doc_id = 0
        with out_path.open("w", encoding="utf-8") as out_f:
            for md_file in md_files:
                doc_id += 1

                base_path = next(
                    p for p in docs_paths
                    if md_file.is_relative_to(p)
                )
                rel_doc_id = str(md_file.relative_to(base_path))
                # logger.debug(f"Начинаем чанкование {rel_doc_id}")

                raw = md_file.read_text(encoding="utf-8")
                text = self.normalize_text(raw)

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

                    section_tokens = self.count_tokens(section_text_with_headers)
                    if combined_section_tokens + section_tokens < 350:
                        combined_section_paths.append(section_path)
                        combined_section_text += section_text_with_headers + "\n\n"
                        combined_section_tokens += section_tokens
                    else:
                        # 2) Режем пред. комбинированную секцию на чанки сплиттером
                        if combined_section_tokens > 0:
                            records, chunk_idx = self.split_section_into_records(
                                section_text=combined_section_text,
                                section_path=combined_section_paths,
                                doc_id=doc_id,
                                rel_doc_id=rel_doc_id,
                                text_splitter=text_splitter,
                                chunk_idx_start=chunk_idx,
                                section_tokens=combined_section_tokens,
                                max_tokens=self.max_tokens,
                            )

                            for record in records:
                                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                total_chunks += 1

                        combined_section_paths = [section_path]
                        combined_section_text = section_text_with_headers + "\n\n"
                        combined_section_tokens = section_tokens

                if combined_section_tokens > 0:
                    records, chunk_idx = self.split_section_into_records(
                        section_text=combined_section_text,
                        section_path=combined_section_paths,
                        doc_id=doc_id,
                        rel_doc_id=rel_doc_id,
                        text_splitter=text_splitter,
                        chunk_idx_start=chunk_idx,
                        section_tokens=combined_section_tokens,
                        max_tokens=self.max_tokens,
                    )

                    for record in records:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1

        logger.info(f"[DONE] Файлов: {len(md_files)}, чанков: {total_chunks}")
        logger.info(f"[OUT] {out_path}")
        logger.info(f"Время потрачено: {time.perf_counter() - t0:.2f}s")

        return total_chunks