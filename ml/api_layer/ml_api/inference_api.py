from typing import Annotated

from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.params import Query
from starlette.requests import Request

from ml.api_layer.di_injections import OCRDep
from ml.api_layer.ml_api.schemas import ImgMetadataSchema, S3SendSchema
from ml.api_layer.ml_api.utils import process_batch_images, save_word_crop, save_result_text
from ml.config import broker
from ml.logger_config import log_event

router = APIRouter(prefix="/inference/ocr", tags=["🔮Inference"])


@router.post('/en')
async def imgs2text(
        imgs: list[UploadFile],
        q_params: Annotated[ImgMetadataSchema, Query(alias='img_ids_list')],
        request: Request,
        model: OCRDep
):
    """
    OCR inference endpoint.
    
    Процесс:
    1. Сохраняет изображения на диск
    2. Запускает OCR inference (батчинг)
    3. Сохраняет вырезки слов и результаты
    4. Отправляет задачу в Kafka для архивации и загрузки в S3

    """
    "Валидация"
    if len(imgs) != len(q_params.img_ids):
        raise HTTPException(status_code=400, detail=f"Number of images ({len(imgs)}) != number of img_ids ({len(q_params.img_ids)})")

    log_event(f"Начали обработку обращений | images: \033[33m{q_params.img_ids}\033[0m", request=request)

    # 1. Сохраняем изображения и загружаем в память
    batch_data = await process_batch_images(imgs, q_params.img_ids)

    # 2. Запускаем OCR inference с БАТЧИНГОМ (детектор + word decoder)
    all_imgs = [img_pil for _, _, img_pil in batch_data]

    # Батчинг: обрабатываем все изображения сразу
    ocr_results = model.forward_pass_batch(all_imgs, return_details=True)

    # 3. Сохраняем вырезки слов и результаты
    results = []

    for (img_id, img_path, img_pil), ocr_result in zip(batch_data, ocr_results):
        # Сохраняем вырезки слов
        for word_idx, word_data in enumerate(ocr_result['words_data']):
            # Вырезаем слово из оригинального изображения
            word_img = model._crop_word(img_pil, word_data['bbox'])

            # Сохраняем вырезку
            save_word_crop(word_img, img_id, word_idx)

        # Сохраняем результат в текстовый файл
        save_result_text(ocr_result['text'], img_id)

        results.append({
            'img_id': img_id,
            'text': ocr_result['text'],
            'word_count': len(ocr_result['words'])
        })

        log_event(f"Processed img_id={img_id}: {len(ocr_result['words'])} words")


    "3. Отправляем задачи в ФОН Kafka"
    s3_bg_event = S3SendSchema(
        img_ids=q_params.img_ids,
        preds_results=[r['text'] for r in results]
    )
    await broker.publish(s3_bg_event, topic='s3-transfer-queue')
    log_event(f"Отправили в фон {len(results)} \033[33m{q_params.img_ids}\033[0m to Kafka", request=request)

    "4. Возвращаем результаты"
    return {
        'success': True,
        'results': results,
        'message': f'Processed {len(results)} images. Archives will be uploaded to S3 in background.'
    }
