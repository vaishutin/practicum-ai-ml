#!/usr/bin/env python3
"""Project verification script for Sprint 7 tasks.

Quick mode (default) performs structural checks and Python syntax validation.
Full mode can additionally run a small end-to-end chunk->index check for Task3
on a tiny subset of the knowledge base (requires extra dependencies).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent


def _print(title: str, ok: bool, details: str = "") -> None:
    status = "OK" if ok else "FAIL"
    msg = f"[{status}] {title}"
    if details:
        msg += f" - {details}"
    print(msg)


def _run(cmd: list[str]) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return True, completed.stdout.strip()
    except subprocess.CalledProcessError as exc:
        return False, (exc.stdout or "").strip()


def check_exists(path: Path, title: str) -> bool:
    ok = path.exists()
    _print(title, ok, str(path))
    return ok


def check_json(path: Path, title: str) -> bool:
    if not path.exists():
        _print(title, False, "file not found")
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        _print(title, True, str(path))
        return True
    except Exception as exc:  # noqa: BLE001 - display reason
        _print(title, False, f"invalid JSON: {exc}")
        return False


def count_md_files(path: Path) -> int:
    return len(list(path.rglob("*.md")))


def run_py_compile(files: Iterable[Path]) -> bool:
    file_list = [str(p) for p in files if p.exists()]
    if not file_list:
        _print("Python syntax check", False, "no files found")
        return False
    ok, out = _run([sys.executable, "-m", "py_compile", *file_list])
    _print("Python syntax check", ok, "py_compile")
    if not ok and out:
        print(out)
    return ok


def check_imports(mods: list[str]) -> bool:
    code = "\n".join([f"import {m}" for m in mods])
    ok, out = _run([sys.executable, "-c", code])
    if not ok and out:
        print(out)
    return ok


def prepare_temp_kb(src_root: Path, temp_root: Path, limit: int = 2) -> list[Path]:
    files = sorted(src_root.rglob("*.md"))[:limit]
    copied: list[Path] = []
    for f in files:
        rel = f.relative_to(src_root)
        dst = temp_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        copied.append(dst)
    return copied


def run_task3_smoke(
    model_name: str,
    allow_downloads: bool,
) -> bool:
    kb_src = ROOT / "Task2" / "knowledge_base"
    if not kb_src.exists():
        _print("Task3 smoke", False, "knowledge_base not found")
        return False

    if not check_imports(["transformers", "langchain"]):
        _print("Task3 smoke", False, "missing transformers/langchain")
        return False

    if not allow_downloads:
        _print(
            "Task3 smoke",
            False,
            "skip (use --allow-downloads to run model-dependent steps)",
        )
        return False

    if not check_imports(["sentence_transformers", "chromadb"]):
        _print("Task3 smoke", False, "missing sentence_transformers/chromadb")
        return False

    temp_root = ROOT / ".tmp_verify"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    (temp_root / "kb").mkdir(parents=True, exist_ok=True)
    (temp_root / "prepared").mkdir(parents=True, exist_ok=True)
    (temp_root / "chroma_db").mkdir(parents=True, exist_ok=True)

    copied = prepare_temp_kb(kb_src, temp_root / "kb", limit=2)
    if not copied:
        _print("Task3 smoke", False, "no KB files to copy")
        return False

    prep_cmd = [
        sys.executable,
        str(ROOT / "Task3" / "scripts" / "prepare_chunks.py"),
        "--docs-path",
        str(temp_root / "kb"),
        "--out-dir",
        str(temp_root / "prepared"),
        "--model-name",
        model_name,
        "--max-tokens",
        "128",
        "--overlap-tokens",
        "16",
    ]
    ok, out = _run(prep_cmd)
    if not ok:
        _print("Task3 smoke", False, "prepare_chunks failed")
        if out:
            print(out)
        return False

    prepared = sorted((temp_root / "prepared").glob("*.jsonl"))
    if not prepared:
        _print("Task3 smoke", False, "no prepared jsonl")
        return False

    build_cmd = [
        sys.executable,
        str(ROOT / "Task3" / "scripts" / "build_index.py"),
        "--chunks-path",
        str(prepared[0]),
        "--chroma-dir",
        str(temp_root / "chroma_db"),
    ]
    ok, out = _run(build_cmd)
    if not ok:
        _print("Task3 smoke", False, "build_index failed")
        if out:
            print(out)
        return False

    _print("Task3 smoke", True, "chunks -> index")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that Sprint 7 project structure and scripts are OK."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run additional smoke checks (may require extra dependencies).",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow model downloads for full checks.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="Embedding model name for Task3 smoke test.",
    )
    args = parser.parse_args()

    print("\n== Structural checks ==")
    check_exists(ROOT / "Project_template.md", "Project template")
    for i in range(1, 8):
        check_exists(ROOT / f"Task{i}" / "README.md", f"Task{i} README")

    print("\n== Task2 checks ==")
    kb = ROOT / "Task2" / "knowledge_base"
    raw = ROOT / "Task2" / "raw_db" / "data"
    if kb.exists():
        _print("knowledge_base md files", True, f"count={count_md_files(kb)}")
    else:
        _print("knowledge_base md files", False, str(kb))
    if raw.exists():
        _print("raw_db md files", True, f"count={count_md_files(raw)}")
    else:
        _print("raw_db md files", False, str(raw))
    check_json(ROOT / "Task2" / "raw_db" / "terms_map.json", "terms_map.json")

    print("\n== Task3/5 scripts syntax ==")
    run_py_compile(
        [
            ROOT / "Task2" / "raw_db" / "replace_terms.py",
            ROOT / "Task3" / "scripts" / "prepare_chunks.py",
            ROOT / "Task3" / "scripts" / "build_index.py",
            ROOT / "Task3" / "scripts" / "search_tests.py",
            ROOT / "Task3" / "scripts" / "delete_index.py",
            ROOT / "Task5" / "scripts" / "prepare_chunks2.py",
        ]
    )

    print("\n== Docker compose files ==")
    for i in (4, 5, 6, 7):
        check_exists(ROOT / f"Task{i}" / "docker-compose.yml", f"Task{i} compose")

    print("\n== Optional artifacts ==")
    check_exists(ROOT / "Task3" / "prepared", "Task3 prepared")
    check_exists(ROOT / "Task3" / "chroma_db" / "chroma.sqlite3", "Task3 chroma")

    if args.full:
        print("\n== Full smoke checks ==")
        run_task3_smoke(args.model_name, args.allow_downloads)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
