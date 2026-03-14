import logging

from app_factory import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag_kb_updater")

# ------------------------ Точка входа
def main() -> None:
    app = create_app()
    logger.info("=========== Starting RAG KB Updater ==============")
    app.start()

if __name__ == "__main__":
    main()
