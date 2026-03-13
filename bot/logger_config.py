import os
import inspect
import json
from datetime import datetime, UTC

import logging
from logging.config import dictConfig

from typing import Literal, Any

from bot.config import env, LOG_DIR






class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "@timestamp": datetime.now(UTC).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "tg-bot",
            "environment": env.app_mode,
            "func": record.__dict__.get('func', 'unknown_function'),
            "location": record.__dict__.get('location', 'unknown_location'),
            "line": record.__dict__.get('line', 0),
        }
        try:
            return json.dumps(log_entry, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            fallback_entry = {
                "@timestamp": datetime.now(UTC).isoformat() + "Z",
                "level": record.levelname,
                "message": str(record.getMessage()),
                "service": "tg-bot",
                "error": f"JSON serialization failed: {str(e)}"
            }
            return json.dumps(fallback_entry, ensure_ascii=False)


lvls = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

logger_settings = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(log_color)s%(levelname)-8s%(reset)s | "
                      "\033[32mD%(asctime)s\033[0m | "
                      "%(cyan)s%(location)s:%(reset)s def %(cyan)s%(func)s%(reset)s(): line - %(cyan)s%(line)d%(reset)s "
                      "%(message)s",
            "datefmt": "%d-%m-%Y T%H:%M:%S",
            "log_colors": {
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red"
            }
        },
        "json": {
            "()": JSONFormatter
        }
    },
    "filters": {},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"
        },
        "json_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": LOG_DIR / "app.log",
            "when": "midnight",
            "backupCount": 30,
            "encoding": "utf8",
            "filters": []
        }
    },
    "loggers": {
        "prod_log": {
            "handlers": ["console", "json_file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

dictConfig(logger_settings)
logger = logging.getLogger('prod_log')


def log_event(event: Any, *args, level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO', **extra):
    event = str(event)
    cur_call = inspect.currentframe()
    outer = inspect.getouterframes(cur_call)[1]
    filename = os.path.relpath(outer.filename)
    func = outer.function
    line = outer.lineno

    message = event % args if args else event

    logger.log(lvls[level], message, extra={
        'location': filename,
        'func': func,
        'line': line,
        **extra
    })