from asyncpg.protocol.record import Record
from fastapi import APIRouter, HTTPException
from starlette.requests import Request

from web.api.tg_routing.ml_service.download_tg_files import tg_file_ids2files
from web.config import AnyAiohttpDep, MLAiohttpDep, env
from web.data.postgres import PgSqlDep
from web.schemas.imgs_schema import TgImgSchema, ImgOcrRateSchema
from web.utils.logger_config import log_event

router = APIRouter(tags=['Telegram Bot OCR'])


@router.post("/images/ocr/inference")
async def inference2tg_bot(body: TgImgSchema, db: PgSqlDep, request: Request, any_aiohttp: AnyAiohttpDep, ml_aiohttp: MLAiohttpDep):
    img_ids = await db.images.save_tg_files_meta(body.tg_id, body.file_ids)
    log_event(f'Сохранили метаданные тг-фотографий | img_ids: \033[33m{img_ids}\033[0m', request=request)

    "0 вставок => Пользователя не существует"
    if len(img_ids) != len(body.file_ids):
        log_event(f'Количества вставок и файлов не совпали | img_ids: \033[31m{img_ids}\033[0m', request=request)
        raise HTTPException(status_code=404, detail='Пользователя с таким tg_id не существует')

    files_form_data, final_img_ids = await tg_file_ids2files(body.tg_file_path_list, img_ids, any_aiohttp)

    "Бульк Запрос на инференс"
    log_event(f'Скачали файлы, несём в MLService на Инференс | f_img_ids: \033[34m{final_img_ids}\033[0m', request=request)
    async with ml_aiohttp.post(
        '/api/v1/inference/ocr/en',
        # params={'img_ids': ','.join(map(str, final_img_ids))},
        params={'img_ids': final_img_ids},
        data=files_form_data,
    ) as resp:
        ml_service_resp = await resp.json()

    "Сохраняем текст предсказаний, предварительный путь к архиву в облаке, меняем статус"
    log_event(f'Получили инференс, кинули фотографии на обновление статуса, сохранение текста | f_img_ids: \033[34m{final_img_ids}\033[0m', request=request)
    await db.images.update_images_status(ml_service_resp.get('results', []))

    log_event(f'Отдаём ответы, обновили данные по фотографиям | f_img_ids: \033[32m{final_img_ids}\033[0m', request=request)
    return ml_service_resp


@router.put("/images/ocr/rate")
async def rate_ocr_inference(body: ImgOcrRateSchema, db: PgSqlDep, request: Request):
    rating = await db.images.rate_ocr_res(body.img_id, body.rate)
    if not rating:
        log_event(f'Оценили несуществующее изображение | img_id: \033[34m{body.img_id}\033[0m; rate: \033[32m{body.rate}\033[0m', request=request, level='ERROR')
        raise HTTPException(status_code=404, detail='Изображение с таким id не найдено')

    log_event(f'Tg user поставил оценку | img_id: \033[33m{body.img_id}\033[0m; rate: \033[32m{body.rate}\033[0m', request=request)
    return {'success': True, 'message': 'Отзыв сохранён'}