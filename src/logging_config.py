import logging
import sys

import structlog

from src.config import LOG_DIR


def configure_logging(log_name: str, level: int = logging.INFO) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
    ]
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False, pad_event=32),
        ],
    )
    console_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(LOG_DIR / f"{log_name}.log", encoding="utf-8")

    # Replace old handlers so re-running a notebook cell does not print every line twice.
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    for handler in (console_handler, file_handler):
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    root_logger.setLevel(level)

    for name in ("ultralytics", "matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
