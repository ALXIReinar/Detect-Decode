import asyncio

from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from aiohttp import ClientSession
from redis.asyncio import Redis


async def catch_img(message: Message, state: FSMContext, aio_http: ClientSession, redis: Redis):
    context = await state.get_data()
    seti = context.get('media_group_controller_set', set())
    photos_videos = context.get('media')

    if message.photo:
        photos_videos.append(message.photo[-1].file_id)

    await state.update_data(media=photos_videos)
    await asyncio.sleep(1)

    if message.media_group_id not in seti:
        "Отрабатывает, если отправлена другая медиагруппа => Уже собрано 10 фотографий. Посылаем на обработку"
        seti.add(message.media_group_id)

