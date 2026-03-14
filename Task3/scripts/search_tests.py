#!/usr/bin/env python
import argparse
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

# ---------- 1. Тестовые запросы (примерные, можно адаптировать) ----------
TEST_QUERIES: List[Dict[str, Any]] = [
    {
        "name": "(Корусант) Вашингтон — Верхние уровни",
        "query": "На каких уровнях расположены пентхаусы на планете Вашингтон?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "places/Корусант.md",
                "section_path": "Вашингтон, Верхние уровни",
            }
        ],
    },
    {
        "name": "Сын Роджера Джоковича",
        "query": "Кто является сыном Роджера Джоковича?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "characters/empire/Вейдер.md",
                "section_path" : "Роджер Джокович"
            }
        ],
    },
    {
        "name": "Джедаи - Светлая сторона Тенниса (Силы)",
        "query": "Какую сторону силы используют джедаи?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "tech/rebels/Светлая_сторона_силы.md",
                "section_path" : "Светлая сторона Тенниса, Светлая сторона Тенниса"
            },
            {
                "rel_doc_id": "tech/rebels/Светлая_сторона_силы.md",
                "section_path" : "Светлая сторона Тенниса, Теннис"
            }
        ],
    },
    {
        "name": "Жена Френсиса Тиафо (Хана Соло) - Серена Органа (Лея Органа)",
        "query": "На ком женился Френсис Тиафо?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "characters/rebels/Хан Соло.md",
                "section_path" : "Френсис Тиафо"
            }
        ],
    },


    {
        "name": "Цветные эскадрильи",
        "query": "Сколько пилотов, чьи имена известны, были в цветных эскадрильях при матче в Париже, а также в какой эскадрилье был корабль Рафаэль Надаль и кто его пилотировал?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Золотая эскадрилья:"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Разбойная эскадрилья (как Красные) (Wilson X-Blade-и и Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Зелёная эскадрилья (Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Синяя эскадрилья (B-wing-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Серая эскадрилья (Wilson Y-Blade-и):"
            },
            {
                "rel_doc_id": "tech/rebels/Тысячелетний_сокол.md",
                "section_path" : "«Рафаэль Надаль»"
            }
        ],
    },
    {
        "name": "Цветные эскадрильи",
        "query": "Собери список известных пилотов из цветных эскадрилий повстанцев в матче при Париже?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Золотая эскадрилья:"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Разбойная эскадрилья (как Красные) (Wilson X-Blade-и и Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Зелёная эскадрилья (Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Синяя эскадрилья (B-wing-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Серая эскадрилья (Wilson Y-Blade-и):"
            }
        ],
    },

    {
        "name": "Серая эскадрилья в Париже",
        "query": "Назови пилотов серой эскадрильи?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Серая эскадрилья (Wilson Y-Blade-и):"
            }
        ],
    },

    {
        "name": "Подсчитай пилотов цветных эскадрилий в Париже",
        "query": "Подсчитай известных пилотов в цветных эскадрильях в Париже?",
        "query_type": "Проверка ХОРОШЕГО запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Золотая эскадрилья:"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Разбойная эскадрилья (как Красные) (Wilson X-Blade-и и Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Зелёная эскадрилья (Wilson A-Blade-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Синяя эскадрилья (B-wing-и):"
            },
            {
                "rel_doc_id": "events/Битва_при_Эндоре.md",
                "section_path" : "Матч при Париже, Флоты, Серая эскадрилья (Wilson Y-Blade-и):"
            }
        ],
    },

    # Хороший запрос (ниже похожий уже плохой) - найдем нужный документ
    {
        "name": "Перехватчик BABOLAT (TIE)",
        "query": "Расскажи, что знаешь про BABOLAT?",
        "query_type": "Проверка ХОРОШЕГО (достаточного по длине и смыслу) запроса - должен быть найден целевой документ",
        "expected_results": [
            {
                "rel_doc_id": "tech/empire/TIE_ln.md",
                "section_path" : "Перехватчик BABOLAT"
            }
        ],
    },

    # ================ ПЛОХИЕ запросы - не должен быть найден нужный документ
    {
        "name": "Перехватчик BABOLAT (TIE)",
        "query": "Что такое BABOLAT?",
        "query_type": "Проверка ПЛОХОГО (слишком короткого, абстрактного) запроса - не должен быть найден целевой документ",
        "unexpected_results": [
            {
                "rel_doc_id": "tech/empire/TIE_ln.md",
                "section_path" : "Перехватчик BABOLAT"
            }
        ],
    },

    {
        "name": "Хан Соло",
        "query": "Кто такой Хан Соло?",
        "query_type": "Проверка ПЛОХОГО (персонаж переименован в RAG) запроса - не должен быть найден целевой документ",
        "unexpected_results": [
            {
                "rel_doc_id": "characters/rebels/Хан Соло.md",
                "section_path" : "Френсис Тиафо"
            }
        ],
    },

    # Этот МОЖЕТ падать - в списке доков ищется Вейдер.md (в числе прочих файлов с персонажами), но LLM дальше ничего не найдет
    # про Вейдера, т.к. старое имя файла в метаданных не будет передаваться в промпте LLM
    {
        "name": "Дарт Вейдер",
        "query": "Кто такой Дарт Вейдер?",
        "query_type": "Проверка ПЛОХОГО (персонаж переименован в RAG) запроса - не должен быть найден целевой документ",
        "unexpected_results": [
            {
                "rel_doc_id": "characters/empire/Вейдер.md",
                "section_path" : "Роджер Джокович"
            }
        ],
    },
]


# ---------- 2. Выбор embedding-модели по имени коллекции и метаданным ----------

def get_model_name_for_collection(client: chromadb.Client, collection_name: str) -> str:
    """
    Достаёт имя embedding-модели из metadata коллекции.

    Ожидается, что коллекция создана с metadata вида:
      metadata={"embedding_model": "BAAI/bge-m3"}
    """

    # Берём коллекцию без embedding_function — только для чтения metadata
    col = client.get_collection(name=collection_name)
    metadata = getattr(col, "metadata", None) or {}

    model_name = (
            metadata.get("embedding_model")
            or metadata.get("model_name")
            or metadata.get("model")  # на всякий случай, если ты назовёшь поле иначе
    )

    if not model_name:
        raise ValueError(
            f"В metadata коллекции '{collection_name}' не найдено поле embedding_model "
            f"(или model_name/model). Добавь его при создании коллекции."
        )

    return model_name

# ---------- 3. Основная логика тестирования запросов ----------

def run_tests(
    chroma_path: str,
    collection_name: str,
    top_k: int = 4,
):
    # 3.1. Инициализация клиента и коллекции
    client = chromadb.PersistentClient(path=chroma_path)
    model_name = get_model_name_for_collection(client, collection_name)
    embedder = SentenceTransformer(model_name)  # model_name взяли из metadata

    collection = client.get_collection(
        name=collection_name,
    )

    total = len(TEST_QUERIES)
    passed = 0

    for idx, test in enumerate(TEST_QUERIES, start=1):
        print("\n" + "=" * 80)
        print(f"[TEST {idx}/{total}] {test['name']}")
        print(f"Query: {test['query']}")
        print(f"Тип проверки (хороший/плохой запрос): {test['query_type']}")

        # 3.2.1 Поиск эмбеддинга запроса (важна нормализация для cosine)
        query_embedding = embedder.encode(
            test["query"],
            normalize_embeddings=True,
        ).tolist()

        # 3.2. Запрос к Chroma
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        print(f"\nTop-{top_k} результатов:")
        for i, (m, d, doc_text, rel_doc_id) in enumerate(
                zip(metadatas, distances, documents, ids), start=1
        ):
            doc_id_meta = m.get("rel_doc_id")
            section_path = m.get("section_path")
            snippet = (doc_text[:120] + "…") if len(doc_text) > 120 else doc_text

            print(f"  #{i}:")
            print(f"    distance    : {d:.4f}")
            print(f"    doc_id      : {doc_id_meta},    chunk_id    : {rel_doc_id}, section_path : {section_path}")
            print(f"    snippet     : {snippet}")

        # 3.3. Проверка expected_results
        test_passed = True
        print("\nОжидаемые результаты:")
        # --- expected_results: должны присутствовать ---
        for expected in test.get("expected_results", []):
            found = any(meta_matches_expected(m, expected) for m in metadatas)

            exp_doc_id = expected.get("rel_doc_id")
            exp_section_path = expected.get("section_path")

            if found:
                print(f"  OK   : найден rel_doc_id={exp_doc_id}, section_path={exp_section_path}")
            else:
                print(f"  FAIL : НЕ найден rel_doc_id={exp_doc_id}, section_path={exp_section_path}")
                test_passed = False


        # --- unexpected_results: НЕ должны присутствовать ---
        for unexpected in test.get("unexpected_results", []):
            found = any(meta_matches_expected(m, unexpected) for m in metadatas)

            un_doc_id = unexpected.get("rel_doc_id")
            un_section_path = unexpected.get("section_path")

            if found:
                print(f"  FAIL : НЕОЖИДАННО найден rel_doc_id={un_doc_id}, section_path={un_section_path}")
                test_passed = False
            else:
                print(f"  OK   : не найден rel_doc_id={un_doc_id}, section_path={un_section_path}")

        if test_passed:
            print("[RESULT] ✅ TEST PASSED")
            passed += 1
        else:
            print("[RESULT] ❌ TEST FAILED")

    print("\n" + "=" * 80)
    print(f"ИТОГО: {passed}/{total} тестов пройдено.")

def _parse_section_path_from_string(s: str) -> list[str]:
    """Парсит строковое представление section_path из metadata Chroma."""
    # убираем внешние [ ]
    inner = s[1:-1]

    # если это [[...]] — убираем ещё одни
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]

    parts = inner.split("], [")

    # "'a', 'b'], ['c'" → "a, b], [c" и убираем кавычки
    return [p.translate({ord("'"): None}) for p in parts]


def meta_matches_expected(m: dict, expected: dict) -> bool:
    exp_doc_id = expected.get("rel_doc_id")
    exp_section_path = expected.get("section_path")

    m_doc_id = m.get("rel_doc_id")
    m_section_path = m.get("section_path")

    if exp_doc_id is not None and m_doc_id != exp_doc_id:
        return False

    if exp_section_path is not None:
        if isinstance(m_section_path, str):
            paths = _parse_section_path_from_string(m_section_path)
        else:
            paths = m_section_path

        if exp_section_path not in paths:
            return False

    return True

# ---------- 4. CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Простой прогон тестовых запросов против коллекции Chroma.",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="chroma_db",
        help="Путь к каталогу с Chroma (по умолчанию ./chroma_db).",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Имя коллекции в Chroma, например BAAI--bge-m3_1024_128.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Сколько документов возвращать для каждого запроса (n_results).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(
        chroma_path=args.chroma_path,
        collection_name=args.collection_name,
        top_k=args.top_k,
    )
