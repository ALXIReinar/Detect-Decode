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
    '.env.ml.prod'
)
load_dotenv(env_files, override=True)
logging.critical(f'\033[35m{env_files}\033[0m | app_mode: \033[33m{os.getenv('APP_MODE')}\033[0m')

WORKDIR = Path(__file__).resolve().parent.parent

"Создаём директорию для логов"
LOG_DIR = WORKDIR / 'ml_logs'
LOG_DIR.mkdir(exist_ok=True, parents=True)

"Директория для хранения картинок/архивов для S3"
S3_DIR = WORKDIR / 's3_samples'
S3_OCR_DIR = S3_DIR / 'ocr'
S3_OCR_DIR.mkdir(parents=True, exist_ok=True)

"Temp директория для успешно загруженных архивов (защита от cron удаления)"
S3_TEMP_DIR = S3_DIR / 'temp'
S3_TEMP_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    device: str = 'cuda' if cuda.is_available() else 'cpu'    
    detector_weights_path: Path
    word_decoder_weights_path: Path
    vocabulary_path: Path

    detector_weights_path_docker: Path
    word_decoder_weights_path_docker: Path
    vocabulary_path_docker: Path

    kafka_host: str
    kafka_port: int
    kafka_host_docker: str
    kafka_port_docker: int
    
    kafka_topic_s3_transfer: str = 's3-transfer-queue'  # Топик для S3 загрузки
    kafka_consumer_group: str = 'ml-ocr-service'
    
    s3_access_key: str
    s3_secret_key: str
    s3_region: str
    s3_endpoint_url: str
    s3_bucket_name: str
    s3_root_cert: str
    s3_root_cert_docker: str
        
    max_det: int  # Максимум детекций для detector в OCRModel
    word_batch_size: int = 32
    
    app_mode: AppMode
    domain: str
    trusted_proxies: set[str] = {"127.0.0.1", "172.18.0.1"}
    uvi_workers: int
    post_processing_responses: bool  # Влияет на производительность API

    model_config = SettingsConfigDict(extra='allow')

@lru_cache
def get_env_vars():
    return Settings()
env = get_env_vars()
logging.critical(f"ML Device: \033[31m{env.device}\033[0m")

"Model Weights Paths"
def get_weights_location(envs: Settings):
    cfg = APP_MODE_CONFIG[envs.app_mode]
    cfg_detector_weights = getattr(envs, cfg['detector_weights_path'])
    cfg_word_decoder_weights = getattr(envs, cfg['word_decoder_weights_path'])

    return cfg_detector_weights, cfg_word_decoder_weights
detector_weights_path, word_decoder_weights_path = get_weights_location(env)

"Kafka"
def get_kafka_broker(envs: Settings) -> KafkaBroker:
    cfg = APP_MODE_CONFIG[envs.app_mode]
    host = getattr(envs, cfg["kafka_host"])
    port = getattr(envs, cfg["kafka_port"])
    bootstrap_servers = f'{host}:{port}'
    
    return KafkaBroker(
        bootstrap_servers,
        graceful_timeout=30.0,
        # logger=logging.getLogger("faststream.kafka"),
        log_level=logging.INFO,
        compression_type='gzip',
    )

broker = get_kafka_broker(env)


"S3 Connection"
def get_s3_config(envs: Settings) -> dict:
    url_config = Config(
        region_name=envs.s3_region,
        s3={'addressing_style': 'virtual'},
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        connect_timeout=10,
        read_timeout=60,
    )

    verify_cert = (
        envs.s3_root_cert_docker 
        if envs.app_mode != AppMode.LOCAL 
        else envs.s3_root_cert
    )
    
    return {
        'aws_access_key_id': envs.s3_access_key,
        'aws_secret_access_key': envs.s3_secret_key,
        'region_name': envs.s3_region,
        'endpoint_url': envs.s3_endpoint_url,
        'config': url_config,
        'verify': verify_cert
    }

s3_config = get_s3_config(env)

@asynccontextmanager
async def async_cloud_session():
    async with async_get_session().create_client('s3', **s3_config) as session:
        yield session