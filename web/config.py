import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# lazy_import???
from torch import cuda

env_files = (
    os.getenv('ENV_FILE') or
    os.getenv('ENV_LOCAL_TEST_FILE') or
    '.env.prod'
)
load_dotenv(env_files, override=True)
logging.critical(f'\033[35m{env_files}\033[0m')

WORKDIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    LOG_DIR: Path = WORKDIR / 'logs'
    dataset_location: str

    device: str = 'cuda' if cuda.is_available() else 'cpu'

    model_config = SettingsConfigDict(extra='allow')


@lru_cache
def get_env_vars():
    return Settings()

env = get_env_vars()
trusted_proxies = {
    '127.0.0.1',
    '172.25.0.1',
}