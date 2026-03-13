from ml.api_layer.ml_api.s3_storage import get_consumer_cloud_session
from ml.api_layer.ml_api.schemas import S3SendSchema
from ml.api_layer.ml_api.utils import create_archive, move_archive_as_succeeded
from ml.config import broker, env
from ml.logger_config import log_event


@broker.subscriber(env.kafka_topic_s3_transfer, group_id=env.kafka_consumer_group)
async def s3_consumer(body: S3SendSchema):
    """
    Консумер для архивации и загрузки в S3.
    
    Процесс:
    1. Создаёт tar.gz архивы из директорий с изображениями
    2. Загружает архивы в S3
    3. Перемещает архивы в S3_TEMP_DIR (защита от cron удаления)
    
    Args:
        body: событие с img_ids и результатами распознавания
    """
    try:
        log_event(f"S3 Consumer: Получено \033[31m{len(body.img_ids)}\033[0m images | начало архивации -> S3")
        
        "1. Создаём архивы для каждого изображения"
        archives = []
        for img_id in body.img_ids:
            "Если один архив провалился, продолжаем"
            try:
                archive_path = create_archive(img_id)
                archives.append((img_id, archive_path))
            except Exception as e:
                log_event(f"Не удалось создать архив | img_id: \033[31m{img_id}\033[0m | {e}", level='ERROR')
                continue
        
        if not archives:
            log_event("Архивы не созданы, скип S3 upload", level='CRITICAL')
            return
        
        "2. Подключение к S3 и загрузка архивов"
        async with get_consumer_cloud_session() as s3:
            for img_id, archive_path in archives:
                try:
                    "Загружаем архив в S3"
                    archive_key = f'ocr/{img_id}.tar.gz' # Должен быть таким же как в cloud_archive_path (inference_api)
                    with open(archive_path, 'rb') as f:
                        await s3.save_file(f, archive_key)
                    
                    log_event(f"Архив в облаке | img_id=\033[32m{img_id}\033[0m; s3_path: \033[33m{archive_key}\033[0m")
                    
                    "3. Перемещаем архив после S3 upload (защита от cron)"
                    move_archive_as_succeeded(archive_path)
                    
                except Exception as e:
                    log_event(f"Не удалось отправить архив в С3 img_id=\033[31m{img_id}\033[0m: {e}", level='ERROR')
                    continue
        
        log_event(f"S3 Consumer: Успешно перенёс архивы в С3 \033[36m{len(archives)}\033[0m", level='WARNING')
        
    except Exception as e:
        log_event(f"S3 Consumer error: {e}", level='CRITICAL')
        raise  # Kafka автоматически сделает retry (настроено в broker)



