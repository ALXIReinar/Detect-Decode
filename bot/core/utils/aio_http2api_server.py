from aiohttp import ClientSession

from bot.logger_config import log_event


class ApiServerConn:
    def __init__(self, aio_http_session: ClientSession):
        self.aio_http_session = aio_http_session

    async def save_user(self, tg_id: int):
        is_created = True  # Пользователи из ТГ регистрируются бесшовно

        "Обращение на Api Server"
        async with self.aio_http_session.post(
                '/api/v1/users/add',
                json={'tg_id': tg_id, 'is_created': is_created}
        ) as resp:
            resp.release() # не нужен ответ, сброс соединения
        log_event(f'Отправили запрос на сохранение пользователя Telegram | tg_id: ...{str(tg_id)[-5:]}')

