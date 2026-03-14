# Задание 6. Автоматическое обновление базы знаний
> Вам нужно полностью автоматизировать процесс обновления базы знаний: скачивание новых документов, преобразование их в чанки, обновление векторного индекса и логирование. Если реально использовать RAG в продукте, то вы неизбежно столкнётесь с тем, что живые данные быстро устаревают.
## Что нужно сделать?
>1. Выберите источник данных.
>   - Это может быть локальная папка (docs/), S3-бакет, публичный Git-репозиторий, RSS-фид, Google Drive, то есть любой реальный или симулированный источник, откуда появляются новые документы. 
>   - Укажите, как будет происходить «подхват» новых файлов.
>2. Напишите скрипт обновления индекса. Скрипт должен:
>   - Сканировать источник и находить новые или изменённые документы.
>   - Разбивать документы на чанки (как в Задании 2).
>   - Генерировать эмбеддинги.
>   - Обновлять векторную БД (добавлять новые чанки, опционально — удалять устаревшие).
>   - Логировать процесс.
>   - Можно использовать Python, bash, make, invoke, airflow — на выбор. Главное — автоматизм.
>3. Настройте периодический запуск.
>   - Варианты:
>     - cron-задача (на Linux/macOS),
>     - системный планировщик (на Windows),
>     - встроенный планировщик в Docker или Kubernetes.
>   - Укажите:
>     - как запускать задачу,
>     - как часто она будет работать (например, каждый день в 6:00),
>     - что будет происходить при ошибке (лог, повтор и т. д.).
>4. Реализуйте логирование. Выведите в лог:
>   - время запуска и завершения,
>   - количество новых чанков,
>   - размер итогового индекса,
>   - ошибки (если были).
>   - Формат может быть простой log.txt, stdout или JSON-файл.
>5. Нарисуйте архитектурную диаграмму. Используя `PlantUML`, покажите:
>   - источник данных → скрипт → чанки → эмбеддинги → индекс,
>   - где хранятся логи,
>   - где и как запускается задача.
>
>**Результат**. По итогу задания у вас должен получиться:
>- Скрипт `update_index.py` или `update.sh` (с пояснением или README).
>- Настроенный cron (или другой планировщик).
>- Рабочее обновление индекса (можно протестировать добавлением нового документа).
>- Диаграмма, которая будет объяснять архитектуру и поток данных.
>- Пример лога: `index updated at 2025-07-17, 3 files added, 0 errors`.

## ADR: общие решения
1. База знаний хранится в локальной папке; в проде это может быть сетевое хранилище компании.
2. Для регулярного обновления индекса `knowledge_base` создано отдельное `py`‑приложение [rag_kb_updater](rag_kb_updater).
   - подход ближе к продовому: настраивается расписание, клиент `chroma`, логирование.
3. Внутри `rag_kb_updater` реализованы:
   - шедулер [Scheduler](rag_kb_updater/services/scheduler.py) на `apscheduler` (cron, ретраи, ограничение конкурентных запусков и т. п.);
   - обновление `.jsonl` с чанками через [Chunker](rag_kb_updater/services/chunker.py) — обёртка над `prepare_chunks.py`;
   - создание/обновление коллекции через [IndexBuilder](rag_kb_updater/services/index_builder.py) — обёртка над `build_index.py` с поддержкой частичного удаления/обновления;
   - [ManifestStore](rag_kb_updater/services/manifest_store.py) — хранение метаданных KB в [manifest.json](rag_kb_updater/_artifacts/manifest.json);
   - [DiffEngine](rag_kb_updater/services/diff_engine.py) — фиксация изменений (added/modified/deleted) по метаданным `manifest.json` (размер, время, хэш);
   - джоба [UpdaterJob](rag_kb_updater/jobs/updater_job.py), оркестрирующая процесс;
   - [RunSummaryWriter](rag_kb_updater/services/run_summary_writer.py) — запись сводок запусков в [runs.jsonl](rag_kb_updater/_runs/runs.jsonl):
     - время, статус, количество изменений, число чанков, размер индекса;
     - размер индекса читается через volume с `chroma_db`; для продового HTTP‑клиента это не идеальный, но приемлемый учебный путь.
4. Логика обновления индекса:
   - по крону ищем изменения в файлах KB;
   - файл чанков пересоздаём всегда (дёшево), а на масштабе можно делать инкрементально;
   - удаляем чанки для `deleted+modified` файлов по метаданным `rel_doc_id`;
   - upsert всех чанков для `modified+added` файлов; оптимизация на будущее — обновление по чанкам, не по файлам;
   - добавлено поле `embedding_id` для корректного апсерта;
   - `embedding_id` = `{имя_файла::номер_чанка}` — стабильный ключ;
   - ранее ключ был вида `{сквозной_номер_файла::номер_чанка}`, что приводило бы к постоянным перестроениям.
5. Нюансы:
   - `Persistent`‑клиент chroma_db не подходит для конкурирующих процессов, поэтому:
     - используется копия [task6/rag-api-safe](rag-api-safe) с HTTP‑клиентом `chroma_db`;
     - `rag_kb_updater` также использует HTTP‑клиент по умолчанию;
     - выделен отдельный контейнер `chroma` для всех клиентов;
     - про каталог индекса:
       - документация предлагала `/chroma/chroma`, но для образа `1.3.5` (и, вероятно, 1.4.1) это приводило к потере данных при рестарте;
       - рабочим оказался путь `/data` (упомянут в [доке](https://docs.trychroma.com/guides/deploy/docker#run-chroma-in-a-docker-container));
       - рабочий конфиг — в [task6/docker-compose.yml](docker-compose.yml).
6. Конфигурация [task6/docker-compose.yml](docker-compose.yml) запускает все контейнеры. В составе:
    - приложение [task6/rag_kb_updater](rag_kb_updater);
    - [task6/rag-api-safe](rag-api-safe) — слегка исправленная копия из задания 5 для удобного конфигурирования клиента chroma;
    - [task5/tg-rag-sw-bot-safe](../task5/tg-rag-sw-bot-safe) — бот из задания 5 без изменений;
    - отдельный контейнер `chroma` для клиент‑серверного режима;
    - каталог [task6/chroma_db](chroma_db_ok2);
    - база знаний [task5/knowledge_base](knowledge_base) — монтируется в `chroma`; копия из задания 2 для удобных тестов изменений;
    - [task6/rag_kb_updater/_artifacts](rag_kb_updater/_artifacts/prepared_chunks) — файл с чанками;
    - [task6/rag_kb_updater/_artifacts/manifest.json](rag_kb_updater/_artifacts/manifest.json) — метаданные базы знаний.
7. Логирование разделено на два уровня:
   - базовое — через логгер в текстовый файл;
   - инкрементное — сводка каждого запуска через [RunSummaryWriter](rag_kb_updater/services/run_summary_writer.py) в jsonl:
     - повышает читаемость и удобство ретраев;
     - облегчает автоматизированную обработку логов.
8. Диаграмма контейнеров RAG.
 ![arch_1-Containers_RAG.svg](schemas/arch_1-Containers_RAG.svg)
9. Диаграмма копонентов `rag-kb-updater`.
 ![arch_2-Components_rag_kb_updater.svg](schemas/arch_2-Components_rag_kb_updater.svg)


## Как запустить в docker или локально?
1. Достаточно пустого каталога [task6/chroma_db](chroma_db). Коллекция будет создана при первом старте обновлятора. Если коллекция непустая, то тоже ок.
2. Получить `OpenAI_API` токен; по аналогии с [task6/rag-api-safe/.env.secrets.example](rag-api-safe/.env.secrets.example) создать файлик `.env.secrets` и добавить туда созданный токен `OpenAI_API`.
3. Создать своего ТГ бота, добавить в него команды:
    - `/change_mode {mode: zero_shot/few_shot/cot}` для смены режима промптинга;
    - `/toggle-safety {safe_prompt: true/false} {safe_prompt: true/false} {safe_prompt: true/false}` для смены режимов безопасности.
4. Добавить токен бота по аналогии с [../task5/tg-rag-sw-bot-safe/.env.secrets.example](../task5/tg-rag-sw-bot-safe/.env.secrets.example) в файлик `.env.secrets`.
5. Загрузить локально нужную `EMBEDDING_MODEL_NAME` [task6/rag-api-safe/.env](rag-api-safe/.env) заранее, чтобы она не скачивалась при каждом запуске контейнера `rag-api-safe`, например:
    ```shell
    pip install --quiet huggingface_hub
    huggingface-cli download BAAI/bge-m3
    ```
6. Старт обоих приложений `docker-compose` из [task6/](../task6) командой ` docker compose -f docker-compose.yml up --build -d`. `rag-api-safe` доступно на 8002 порту.

## Результаты
### Общие результаты
- Настроено обновление индекса KB и протестировано в связке с ботом.
- Фрагмент `runs.jsonl`: видно первичное создание коллекции (`added` = 30 документов), затем частичные обновления, удаления, добавления и «пустые» прогоны:
```json lines
{"status": "success", "error": null, "files": {"added": 30, "modified": 0, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 99, "db_deleted": 0}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564180006, "finished_at_ms": 1767564286129, "duration_ms": 106123}
{"status": "success", "error": null, "files": {"added": 0, "modified": 0, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 0, "db_deleted": 0}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564300009, "finished_at_ms": 1767564300167, "duration_ms": 158}
{"status": "success", "error": null, "files": {"added": 0, "modified": 1, "deleted": 2}, "chunks": {"prepared": 95, "db_upserted": 3, "db_deleted": 3}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564360012, "finished_at_ms": 1767564362947, "duration_ms": 2935}
{"status": "success", "error": null, "files": {"added": 2, "modified": 0, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 4, "db_deleted": 0}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564420015, "finished_at_ms": 1767564424117, "duration_ms": 4102}
{"status": "success", "error": null, "files": {"added": 0, "modified": 1, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 3, "db_deleted": 1}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564507472, "finished_at_ms": 1767564511066, "duration_ms": 3594}
{"status": "success", "error": null, "files": {"added": 0, "modified": 0, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 0, "db_deleted": 0}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564540008, "finished_at_ms": 1767564540144, "duration_ms": 136}
{"status": "success", "error": null, "files": {"added": 0, "modified": 0, "deleted": 0}, "chunks": {"prepared": 99, "db_upserted": 0, "db_deleted": 0}, "db_index": {"size_bytes": 5200036}, "started_at_ms": 1767564600012, "finished_at_ms": 1767564600130, "duration_ms": 118}

```
- пример диалога бота: файл про Алькараса (Палпатин) был удалён, затем возвращён; часть знаний была в других документах, но дата рождения — только в ключевой статье:
    ```markdown
    Родившийся в 82 ДОКК на планете Набу в аристократическом доме Алькарасов ....
    ```
  ![kb_update_bot_dialog.png](images/kb_update_bot_dialog.png)
