from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from rag.config import settings


def get_logger(name: str = "rag") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if settings.debug_mode else logging.INFO)
    Path(settings.log_file).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, payload: Dict[str, Any]) -> None:
    logger.info("%s %s", event, json.dumps(payload, ensure_ascii=True, default=str))
