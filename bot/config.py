import logging
import os
from functools import lru_cache
from pathlib import Path

from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


from bot.env_modes import AppMode, APP_MODE_CONFIG

env_files = (
    os.getenv('ENV_FILE') or
    os.getenv('ENV_LOCAL_TEST_FILE') or
    '.env.bot.prod'
)
load_dotenv(env_files, override=True)
logging.critical(f'\033[35m{env_files}\033[0m | app_mode: \033[33m{os.getenv('APP_MODE')}\033[0m')

WORKDIR = Path(__file__).resolve().parent.parent

"Создаём директорию для логов"
LOG_DIR = WORKDIR / 'bot_logs'
LOG_DIR.mkdir(exist_ok=True, parents=True)


class Settings(BaseSettings):
    bot_token: str
    api_server_url: str
    api_server_url_docker: str
    admin_id: int

    app_mode: AppMode
    domain: str
    auth_api_service_secret: str

    model_config = SettingsConfigDict(extra='allow')

@lru_cache
def get_env_vars():
    return Settings()
env = get_env_vars()


api_base_url = getattr(env, APP_MODE_CONFIG[env.app_mode]['api_server_url'])


"Bot"
bot = Bot(token=env.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))