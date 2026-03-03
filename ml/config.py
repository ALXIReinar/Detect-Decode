import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from botocore.config import Config
from aiobotocore.session import get_session as async_get_session
from faststream.kafka import KafkaBroker
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from torch import cuda

from ml.env_modes import AppMode, APP_MODE_CONFIG

env_files = (
    os.getenv('ENV_FILE') or
    os.getenv('ENV_LOCAL_TEST_FILE') or
    '.env.prod'
)
load_dotenv(env_files, override=True)
logging.critical(f'\033[35m{env_files}\033[0m')

WORKDIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    device: str = 'cuda' if cuda.is_available() else 'cpu'
    detector_weights_path: Path
    word_decoder_weights_path: Path

    kafka_host: str
    kafka_port: int
    kafka_host_docker: str
    kafka_port_docker: int

    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    s3_endpoint_url: str
    s3_bucket_name: str
    s3_root_cert: str
    s3_root_cert_docker: str

    app_mode: AppMode
    auth_service_secret: str
    trusted_proxies: set[str] = {'127.0.0.1', '172.18.0.1'}
    uvi_workers: int
    post_processing_responses: bool # влияет на производительность апи
    model_config = SettingsConfigDict(extra='allow')


@lru_cache
def get_env_vars():
    return Settings()

env = get_env_vars()


"Kafka"
def get_kafka_settings(envs: Settings):
    cfg = APP_MODE_CONFIG[envs.app_mode]
    host = getattr(envs, cfg["kafka_host"])
    port = getattr(envs, cfg["kafka_port"])
    return f'{host}:{port}'
broker = KafkaBroker(get_kafka_settings(env))


"S3 Conn"
url_config = Config(
    region_name='ru-7',
    s3={'addressing_style': 'virtual'}
)
s3_config =  {
    'aws_access_key_id': env.s3_access_key,
    'aws_secret_access_key': env.s3_secret_key,
    'region_name': env.s3_region,
    'endpoint_url': env.s3_endpoint_url,
    'config': url_config,
    'verify': env.s3_root_cert_docker if env.dockerized else env.s3_root_cert
}
@asynccontextmanager
async def async_cloud_session():
    async with async_get_session().create_client('s3', **s3_config) as session:
        yield session