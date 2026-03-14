import logging
import time

from app_factory import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------ Точка входа
def main() -> None:
    app = create_app()
    logger.info("=========== Starting RAG Monitor ==============")
    app.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Shutting down RAG Monitor")

if __name__ == "__main__":
    main()
