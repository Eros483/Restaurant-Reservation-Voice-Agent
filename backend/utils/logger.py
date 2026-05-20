# ----- central logging utility @ backend/utils/logger.py -----

import logging
import os
from datetime import datetime
from typing import Optional

from config import settings

os.makedirs(settings.log_dir, exist_ok=True)

LOG_FILE = os.path.join(
    settings.log_dir, f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s-%(levelname)s-%(name)s-%(message)s",
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
)

console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
console_handler.setFormatter(
    logging.Formatter("%(asctime)s-%(levelname)s-%(name)s-%(message)s")
)

_logger_cache: dict[str, logging.Logger] = {}


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = settings.app_name

    if name not in _logger_cache:
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        if not logger.handlers:
            logger.addHandler(console_handler)
        _logger_cache[name] = logger

    return _logger_cache[name]


logger = get_logger()
