import logging
from datetime import datetime
from logging import Logger
from pathlib import Path

from config import LOG_DIR


def get_logger(name: str = "rag_app") -> Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    session_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_log_file = LOG_DIR / f"session_{session_time}.log"

    logger = logging.getLogger(f"{name}_{session_time}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(session_log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.info("session_log_file=%s", session_log_file)

    return logger