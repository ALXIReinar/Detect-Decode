import os
from typing import Annotated

from botocore.exceptions import ClientError
from fastapi import HTTPException
from fastapi.params import Depends
from starlette.datastructures import UploadFile

from ml.config import async_cloud_session, env
from ml.logger_config import log_event


class S3StorageAsync:
    def __init__(self, _cloud_session, bucket_name):
        self.bucket = bucket_name
        self.session = _cloud_session


    async def save_file(self, uploaded_file, file_path: str):
        log_event('Файл сохраняется в С3 | file: %s', file_path)
        if isinstance(uploaded_file, UploadFile):
            f = uploaded_file.file
            content_length = uploaded_file.size
        else:
            f = uploaded_file
            f.seek(0, os.SEEK_END)
            content_length = f.tell()
            f.seek(0) # Проверка на file-like объект, а не bytes | str
        log_event('Тип файла: %s; size: %s; filename: %s', type(f), content_length, file_path)
        try:
            await self.session.put_object(
                Bucket=self.bucket,
                Key=file_path,
                Body=f,
                ContentLength=content_length
            )
        except Exception as e:
            log_event(f'Ошибка сохранения файла в С3! | {e}', level='CRITICAL')
            raise HTTPException(status_code=504, detail={'success': False, 'message': 'Ошибка загрузки файла'})
        log_event('Записан Файл в S3 \033[35m(Уровень Облака)\033[0m | file: %s', file_path)


    async def set_presigned_url(self, file_key: str, ttl: int = 3600):
        params = {
            'Bucket': self.bucket,
            'Key': file_key
        }
        presigned_url = await self.session.generate_presigned_url(
            'get_object',
            Params=params,
            ExpiresIn=ttl
        )
        log_event('Выдан юрл на s3-объект: %s', file_key)
        return presigned_url


    async def ping_object(self, file_key: str, bucket_name: str):
        try:
            await self.session.head_object(Key=file_key, Bucket=bucket_name)
            log_event('Файл в S3 | %s', file_key, level='WARNING')
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                log_event('Файла нет в S3: %s', file_key)
            else:
                log_event(f'Произошло нечто иное | file_key: {file_key} | Exception: {e.response}', level='CRITICAL')
        return False


async def set_async_cloud_session():
    async with async_cloud_session() as cloud:
        yield S3StorageAsync(cloud, env.s3_bucket_name)

S3Dep = Annotated[S3StorageAsync, Depends(set_async_cloud_session)]
