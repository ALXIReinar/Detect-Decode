import asyncio

from aiogram.types import Message
from redis.asyncio import Redis

from bot.config import bot, env
from bot.core.utils.aio_http2api_server import ApiServerConn
from bot.core.utils.anything import RedisKeys, post_processing_text
from bot.core.utils.keyboards import inference_feedback
from bot.core.utils.rate_limiter import rate_limit
from bot.logger_config import log_event



@rate_limit(env.user_req_limit, env.user_req_window_seconds)
async def catch_imgs(message: Message, redis: Redis, aio_http: ApiServerConn):
    """
    Для обработки документов(фото-файлов) требуется доработать логику для выбора между message.photo и message.document

    """
    tg_id = message.from_user.id

    "Если фото только одно"
    if not message.media_group_id:
        "Тот же лок, иначе по одному сообщению можно отправлять +inf раз"
        lock_key = RedisKeys.media_lock(message.from_user.id)
        is_first = await redis.set(lock_key, "1", ex=86_400, nx=True)  # 1 day
        if not is_first:
            log_event(f'Попытка отправить Фото во время инференса!!! | tg_id: \033[35m...{str(tg_id)[-5:]}\033[0m', level='WARNING')
            return

        await message.answer('Обработка...')

        file_id = message.photo[-1].file_id
        file_path_list = await bot.get_file(file_id)
        text_from_images = await aio_http.imgs2inference(tg_id, [file_id], [file_path_list.file_path])
        log_event(f'Получили инференс от АпиСервера | res_len: \033[34m{len(text_from_images)}\033[0m', level='WARNING')
        img_pred = text_from_images[0]
        await bot.delete_message(message.chat.id, message.message_id + 1)

        normalized_text = post_processing_text(img_pred['text'], lang='ru')
        await message.answer(
            f'Фото <b>№{img_pred["img_id"]}</b> | Всего слов: <b>{img_pred["word_count"]}</b>\n\n{normalized_text}',
            reply_markup=inference_feedback(img_pred['img_id'])
        )
        await redis.delete(lock_key)
        log_event(f'Отдали ответ в бота, сняли лок | tg_id: \033[35m...{str(tg_id)[-5:]}\033[0m')
        return
    
    "Редис-ключи"
    media_group_key = RedisKeys.media_group(message.media_group_id)
    lock_key = RedisKeys.media_lock(message.media_group_id)
    
    "Добавляем file_id в Redis список для этой media_group"
    await redis.rpush(media_group_key, message.photo[-1].file_id)
    await redis.expire(media_group_key, 60)  # TTL 60 секунд
    
    "Атомарная проверка: только первое сообщение установит lock"
    is_first = await redis.set(lock_key, "1", ex=86_400, nx=True) # 1 day
    if is_first:
        await asyncio.sleep(1.5) # Задержка для сбора всех фото из медиа группы

        media_list = await redis.lrange(media_group_key, 0, -1)
        file_path_list = [
            (await bot.get_file(file_id)).file_path # собираем file_path, чтобы скачать файлы в ApiService
            for file_id in media_list
        ]

        await message.answer('Обработка...')
        
        "Отправляем в АпиСервер"
        text_from_images = await aio_http.imgs2inference(tg_id, media_list, file_path_list)
        log_event(f'Получили инференс от АпиСервера | res_len: \033[34m{len(text_from_images)}\033[0m', level='WARNING')
        await bot.delete_message(message.chat.id, message.message_id + len(media_list)) # Удаляем сообщение "Обработка..."

        "Фоллбек если нет сообщений для отправки"
        if len(text_from_images) == 0:
            log_event(f'Пользователь не получил предсказаний!!! media_list: {media_list}', level='CRITICAL')
            await message.answer('К сожалению, не удалось получить ответ. Мы уже решаем эту проблему⚠️⚙️')
            await bot.send_message(env.admin_id, f'Пользователь <b>{message.from_user.id}</b>\nНе получил инференс. file_ids:\n{'\n'.join(media_list)}')

        "Постпроцессинг + Отправка"
        for img_pred in text_from_images:
            normalized_text = post_processing_text(img_pred['text'], lang='ru')
            await message.answer(
                f'Фото <b>№{img_pred["img_id"]}</b> | Всего слов: <b>{img_pred["word_count"]}</b>\n\n{normalized_text}',
                reply_markup=inference_feedback(img_pred['img_id'])
            )
        
        "Очищаем данные"
        await redis.delete(media_group_key, lock_key)
        log_event(f'Отдали ответ в бота, сняли лок | \033[35m{media_group_key}\033[0m')
