import asyncio

from aiogram.types import Message
from redis.asyncio import Redis

from bot.config import bot
from bot.core.utils.aio_http2api_server import ApiServerConn
from bot.core.utils.anything import RedisKeys, post_processing_text
from bot.core.utils.keyboards import inference_feedback


async def catch_imgs(message: Message, aio_http: ApiServerConn, redis: Redis):
    tg_id = message.from_user.id

    "Если фото только одно"
    if not message.media_group_id:
        "Тот же лок, иначе по одному сообщению можно отправлять +inf раз"
        lock_key = RedisKeys.media_lock(message.from_user.id)
        is_first = await redis.set(lock_key, "1", ex=86_400, nx=True)  # 1 day
        if not is_first:
            return

        await message.answer('Обработка...')
        text_from_images = await aio_http.imgs2inference(tg_id, [message.photo[-1].file_id])
        img_pred = text_from_images[0]
        await bot.delete_message(message.chat.id, message.message_id + 1)

        normalized_text = post_processing_text(img_pred['text'], lang='ru')
        await message.answer(
            f'Фото <b>№{img_pred["img_id"]}</b> | Всего слов: <b>{img_pred["word_count"]}</b>\n\n{normalized_text}',
            reply_markup=inference_feedback(img_pred['img_id'])
        )
        await redis.delete(lock_key)
        return
    
    "Редис-ключи"
    media_group_key = f"media_group:{message.media_group_id}"
    lock_key = RedisKeys.media_lock(message.media_group_id)
    
    "Добавляем file_id в Redis список для этой media_group"
    await redis.rpush(media_group_key, message.photo[-1].file_id)
    await redis.expire(media_group_key, 60)  # TTL 60 секунд
    
    "Атомарная проверка: только первое сообщение установит lock"
    is_first = await redis.set(lock_key, "1", ex=86_400, nx=True) # 1 day
    if is_first:
        await asyncio.sleep(1.5) # Задержка для сбора всех фото из медиа группы

        media_list = await redis.lrange(media_group_key, 0, -1)

        await message.answer('Обработка...')
        
        "Отправляем в АпиСервер"
        text_from_images = await aio_http.imgs2inference(tg_id, media_list)
        await bot.delete_message(message.chat.id, message.message_id + len(media_list)) # Удаляем сообщение "Обработка..."

        "Постпроцессинг + Отправка"
        for img_pred in text_from_images:
            normalized_text = post_processing_text(img_pred['text'], lang='ru')
            await message.answer(
                f'Фото <b>№{img_pred["img_id"]}</b> | Всего слов: <b>{img_pred["word_count"]}</b>\n\n{normalized_text}',
                reply_markup=inference_feedback(img_pred['img_id'])
            )
        
        "Очищаем данные"
        await redis.delete(media_group_key, lock_key)