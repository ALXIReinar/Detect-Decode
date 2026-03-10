import aiohttp
from asyncpg import Record

from web.config import AnyAiohttpDep, env


async def tg_file_ids2files(file_path_list: list[str], img_ids: list[Record], aio_http_obj: AnyAiohttpDep):
    """
    Скачиваем файлы с Телеграма, возвращаем File-Like без сохранения файлов на диск
    """
    form_data = aiohttp.FormData()
    final_img_ids = []
    "Скачиваем Файлы с Тг-БотАпи"
    for file_path, img_id in zip(file_path_list, img_ids):
        img_id = img_id['id']

        "Запрос-соединение на каждый файл"
        async with aio_http_obj.get(
                env.telegram_download_file_endpoint.format(bot_token=env.bot_token, file_path=file_path),
        ) as resp:
            if resp.status != 200:
                "Скип если не удалось получить файл от Тг"
                resp.release()
                continue

            filename = f'{img_id}.jpg'
            content = await resp.read()
            form_data.add_field('imgs', content, filename=filename, content_type='image/jpeg') # ВАЖНО! imgs - параметр в API MLService
            final_img_ids.append(img_id)

    return form_data, final_img_ids