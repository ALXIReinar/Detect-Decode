from aiohttp import ClientSession, ClientResponse
from aiohttp.web_response import Response

from bot.logger_config import log_event


class ApiServerConn:
    def __init__(self, aio_http_session: ClientSession):
        self.aio_http_session = aio_http_session

    async def save_user(self, tg_id: int, first_name: str, last_name: str):
        is_created = True  # Пользователи из ТГ регистрируются бесшовно

        "Обращение на Api Server"
        async with self.aio_http_session.post(
                '/api/v1/telegram/users/add',
                json={'tg_id': tg_id, 'first_name': first_name, 'last_name': last_name, 'is_created': is_created}
        ) as resp:
            resp.release() # не нужен ответ, сброс соединения
        log_event(f'Отправили запрос на сохранение пользователя Telegram | tg_id: ...{str(tg_id)[-5:]}')


    async def imgs2inference(self, tg_id: int, media_list: list[str], file_path_list: list[str]):
        """
        Запрос на Инференс по изображениям
        Тело ответа:
        {
            'success': True,
            'results': [
                {
                    'img_id': img_id,
                    'text': 'Hello World!',
                    'word_count': 2
                }
            ],
            'message': '123321'
        }
        """
        log_event(f'Отправили на инференс file_ids: \033[33m{len(media_list)}\033[0m')
        async with self.aio_http_session.post(
                '/api/v1/telegram/images/ocr/inference',
                json={'tg_id': tg_id, 'file_ids': media_list, 'file_path_list': file_path_list}
        ) as resp:
            resp.raise_for_status()
            resp = await resp.json()

        results = resp['results']
        return results


    async def rate_img_inference(self, img_id: int | str, rate: int | str):
        """Пользователь ставит оценку предсказанию модели"""
        async with self.aio_http_session.put(
            '/api/v1/telegram/images/ocr/rate',
            json={'img_id': img_id, 'rate': rate}
        ) as resp:
            resp.release() # не нужен ответ, сброс соединения
        log_event(f'Оценили изображение | img_id: \033[35m{img_id}\033[0m; rate: \033[32m{rate}\033[0m')
