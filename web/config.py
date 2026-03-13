import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from aiohttp import ClientSession
from fastapi import Depends
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from starlette.requests import Request

from web.env_modes import AppMode, APP_MODE_CONFIG

env_files = (
    os.getenv('ENV_FILE') or
    os.getenv('ENV_LOCAL_TEST_FILE') or
    '.env.api.prod'
)
load_dotenv(env_files, override=True)
logging.critical(f'\033[35m{env_files}\033[0m | app_mode: \033[32m{os.getenv('APP_MODE')}\033[0m')

"Создаём директории"
WORKDIR = Path(__file__).resolve().parent.parent

LOG_DIR = WORKDIR / 'web_logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    bot_token: str
    telegram_download_file_endpoint: str = 'https://api.telegram.org/file/bot{bot_token}/{file_path}'

    pg_user: str
    pg_password: str
    pg_max_connections: int
    pg_db: str
    pg_port: int
    pg_host: str
    pg_port_docker: int
    pg_host_docker: str

    redis_password: str
    redis_host: str
    redis_port: int
    redis_port_docker: int
    redis_host_docker: str

    ml_server_url_docker: str
    ml_server_url: str
    uvi_workers: int
    post_processing_responses: bool
    app_mode: AppMode
    trusted_proxies: set[str] = {'127.0.0.1', '172.18.0.1', '172.18.0.9'}
    model_config = SettingsConfigDict(extra='allow')
    domain: str


@lru_cache
def get_env_vars():
    return Settings()

env = get_env_vars()


"PostgreSQL"
def get_pg_settings(envs: Settings):
    cfg = APP_MODE_CONFIG[envs.app_mode]
    host = getattr(envs, cfg["pg_host"])
    port = getattr(envs, cfg["pg_port"])

    return {"host": host, "port": port}

pool_settings = dict(
    user=env.pg_user,
    database=env.pg_db,
    password=env.pg_password,
    **get_pg_settings(env),
    command_timeout=60,
    max_size=env.pg_max_connections # connections on pool
)


"ML AioHttp"
def get_ml_settings(envs: Settings):
    cfg = APP_MODE_CONFIG[envs.app_mode]
    base_url = getattr(envs, cfg["ml_server_url"])
    return {"base_url": base_url}
ml_aiohttp_settings = get_ml_settings(env)

async def get_ml_aiohttp_session(request: Request) -> ClientSession:
    return request.app.state.ml_aiohttp

MLAiohttpDep = Annotated[ClientSession, Depends(get_ml_aiohttp_session)]


"AioHttp для микроопераций"
async def get_any_aiohttp(request: Request) -> ClientSession:
    return request.app.state.any_aiohttp

AnyAiohttpDep = Annotated[ClientSession, Depends(get_any_aiohttp)]