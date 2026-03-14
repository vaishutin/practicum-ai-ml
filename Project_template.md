# Спринт 7 Создание AI/ML чат-бота

- [Задание 1](./Task1/README.md)
- [Задание 2](./Task2/README.md)
- [Задание 3](./Task3/README.md)
- [Задание 4](./Task4/README.md)
- [Задание 5](./Task5/README.md)
- [Задание 6](./Task6/README.md)
- [Задание 7](./Task7/README.md)

## Проверка проекта
Результат локального запуска можно посмотреть в [script-output.logs](./script-output.logs)

Быстрая проверка структуры и скриптов:
```bash
.venv/bin/python verify_project.py
```

Проверка RAG в Docker с ключом OpenAI (ключ передаётся аргументом):
```bash
.venv/bin/python verify_project.py --openai-key "YOUR_OPENAI_API_KEY"
```

Полная проверка со скачиванием моделей:
```bash
.venv/bin/python verify_project.py --full --allow-downloads --openai-key "YOUR_OPENAI_API_KEY"
```
