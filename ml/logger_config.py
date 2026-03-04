import os
import inspect
import json
from pathlib import Path
from datetime import datetime, UTC

import logging
from logging.config import dictConfig

from typing import Literal, Any

from starlette.requests import Request

from ml.config import env, LOG_DIR



def get_client_ip(request: Request):
    xff = request.headers.get('X-Forwarded-For')
    ip = xff.split(',')[0].strip() if (
            xff and request.client.host in env.trusted_proxies
    ) else request.client.host
    return ip



class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "@timestamp": datetime.now(UTC).isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "service": "ml-server",
            "environment": env.app_mode,
            "method": record.__dict__.get('method', ''),
            "url": str(record.__dict__.get('url', '')),
            "func": record.__dict__.get('func', 'unknown_function'),
            "location": record.__dict__.get('location', 'unknown_location'),
            "line": record.__dict__.get('line', 0),
        }

        extra_fields = [
            'http_status', 'response_time', 'cpu_percent', 'memory_percent',
            'memory_used_mb', 'memory_total_mb', 'metric_type'
        ]
        for key in extra_fields:
            log_entry[key] = record.__dict__.get(key, '')

        try:
            return json.dumps(log_entry, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            fallback_entry = {
                "@timestamp": datetime.now(UTC).isoformat() + "Z",
                "level": record.levelname,
                "message": str(record.getMessage()),
                "service": "ml-server",
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
                      "\033[34m%(method)s\033[0m \033[36m%(url)s\033[0m | "
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


def log_event(event: Any, *args, request: Request = None,
              level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO', **extra):
    event = str(event)
    cur_call = inspect.currentframe()
    outer = inspect.getouterframes(cur_call)[1]
    filename = os.path.relpath(outer.filename)
    func = outer.function
    line = outer.lineno

    meth, url = '', ''
    if isinstance(request, Request):
        meth, url = request.method, str(request.url.path)

    message = event % args if args else event

    logger.log(lvls[level], message, extra={
        'method': meth,
        'location': filename,
        'func': func,
        'line': line,
        'url': url,
        **extra
    })