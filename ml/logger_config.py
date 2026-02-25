import os
import inspect

import logging
from logging.config import dictConfig
from typing import Literal

from ml.config import env


"Init Log dirs"
def create_log_dirs():
    env.LOG_DIR.mkdir(exist_ok=True)

    debug, info_warn, err = env.LOG_DIR / 'debug', env.LOG_DIR / 'info_warning', env.LOG_DIR / 'error'

    debug.mkdir(exist_ok=True, parents=True)
    info_warn.mkdir(exist_ok=True, parents=True)
    err.mkdir(exist_ok=True, parents=True)
create_log_dirs()



class InfoWarningFilter(logging.Filter):
    def logger_filter(self, log):
        return log.levelno in (logging.INFO, logging.WARNING, logging.ERROR)

class ErrorFilter(logging.Filter):
    def logger_filter(self, log):
        return log.levelno == logging.CRITICAL

class DebugFilter(logging.Filter):
    def logger_filter(self, log):
        return log.levelno == logging.DEBUG

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
        "no_color": {
            "()": "colorlog.ColoredFormatter",
            "format": "%(levelname)-8s | "
                      "D%(asctime)s | "
                      "%(location)s: def %(func)s(): line - %(line)d "
                      "%(message)s",
            "datefmt": "%d-%m-%Y T%H:%M:%S",
            "log_colors": {
                "DEBUG": "white",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red"
            }
        }
    },
    "filters": {
        "info_warning_error_filter": {
            "()": InfoWarningFilter,
        },
        "error_filter": {
            "()": ErrorFilter,
        },
        "debug_filter": {
            "()": DebugFilter,
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG"
        },
        "debug_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "no_color",
            "filename": env.LOG_DIR / "debug" / "app.log",
            "when": "midnight",
            "backupCount": 60,
            "encoding": "utf8",
            "filters": ["debug_filter"]
        },
        "info_warning_errors_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "no_color",
            "filename": env.LOG_DIR / "info_warning" / "app.log",
            "when": "midnight",
            "backupCount": 60,
            "encoding": "utf8",
            "filters": ["info_warning_error_filter"]
        },
        "critical_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "CRITICAL",
            "formatter": "no_color",
            "filename": env.LOG_DIR / "error" / "app.log",
            "when": "midnight",
            "backupCount": 180,
            "encoding": "utf8",
            "filters": ["error_filter"]
        }
    },
    "loggers": {
        "prod_log": {
            "handlers": ["console", "info_warning_errors_file", "critical_file", "debug_file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

logging.config.dictConfig(logger_settings)
logger = logging.getLogger('prod_log')


def log_event(event: str, *args, level: Literal['DEBUG','INFO','WARNING','ERROR','CRITICAL'] = 'INFO'):
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
    })
