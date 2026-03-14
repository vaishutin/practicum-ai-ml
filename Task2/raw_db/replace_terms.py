#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Iterable, List, Tuple


PatternReplacements = List[Tuple[re.Pattern, str]]


def load_terms(path: str) -> PatternReplacements:
    """Читаем файл с терминами и готовим паттерны:
    1) фразы
    2) точные одиночные слова
    3) одиночные слова с окончаниями (Дарт* → Пит*)
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # сортируем по длине ключа: более длинные — первыми
    items = sorted(data.items(), key=lambda kv: len(kv[0]), reverse=True)

    phrase_exact = []      # "после битвы при Явине"
    word_exact = []        # "Дарт"
    word_with_suffix = []  # "Дарт*" → "Пит*"

    for old, new in items:
        if ' ' in old:
            # фраза — только точное совпадение
            pattern = re.compile(r'\b' + re.escape(old) + r'\b',
                                 re.IGNORECASE)
            phrase_exact.append((pattern, new))
        else:
            # одиночное слово: точное совпадение
            pattern_word = re.compile(r'\b' + re.escape(old) + r'\b',
                                      re.IGNORECASE)
            word_exact.append((pattern_word, new))

            # и вариант с окончанием: Дарт, Дарта, Дарту...
            pattern_suffix = re.compile(
                r'\b' + re.escape(old) + r'(\w+)\b', re.IGNORECASE
            )
            word_with_suffix.append((pattern_suffix, new))

    # порядок важен:
    #   1) фразы
    #   2) точные слова
    #   3) слова с окончаниями
    return phrase_exact + word_exact + word_with_suffix


def replace_terms(text: str, patterns: Iterable[tuple[re.Pattern, str]]) -> str:
    """Заменяем термины, сохраняя регистр и (для слов) окончания."""

    def make_replacer(new):
        def repl(match):
            orig = match.group(0)

            # если есть группа с окончанием (для паттерна с (\w+))
            if match.lastindex and match.lastindex >= 1:
                suffix = match.group(match.lastindex)
                result = new + suffix
            else:
                result = new

            # поднимаем первую букву, если исходное слово было с заглавной
            if orig and orig[0].isupper():
                return result[:1].upper() + result[1:]
            return result

        return repl

    for pattern, new in patterns:
        text = pattern.sub(make_replacer(new), text)
    return text


def copy_with_replacements(src_root: str, dest_root: str, patterns: PatternReplacements) -> None:
    """Копируем каталог, во всех текстовых файлах заменяем термины."""
    for dirpath, dirnames, filenames in os.walk(src_root):
        rel = os.path.relpath(dirpath, src_root)
        dest_dir = os.path.join(dest_root, rel) if rel != '.' else dest_root
        os.makedirs(dest_dir, exist_ok=True)

        for name in filenames:
            src_file = os.path.join(dirpath, name)
            dest_file = os.path.join(dest_dir, name)

            # пробуем читать как текст; иначе копируем как бинарный файл
            try:
                with open(src_file, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(src_file, 'rb') as fsrc, open(dest_file, 'wb') as fdst:
                    fdst.write(fsrc.read())
                continue

            new_text = replace_terms(text, patterns)
            with open(dest_file, 'w', encoding='utf-8') as f:
                f.write(new_text)


def main():
    parser = argparse.ArgumentParser(description="Замена терминов по terms.json")
    parser.add_argument('--src', required=True, help='Исходный каталог с документами')
    parser.add_argument('--dest', required=True, help='Каталог для результата')
    parser.add_argument('--terms', required=True, help='Файл terms.json')
    args = parser.parse_args()

    patterns = load_terms(args.terms)
    os.makedirs(args.dest, exist_ok=True)
    copy_with_replacements(args.src, args.dest, patterns)


if __name__ == '__main__':
    main()
