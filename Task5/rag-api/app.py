import os
import logging

from app_factory import create_app

from pygments.lexers import blueprint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-app")

# ------------------------ Точка входа
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
    logger.info("=========== rag-api started ==============")