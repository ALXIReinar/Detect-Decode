import secrets
import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Send, Scope

from ml.config import env
from ml.logger_config import log_event


class ASGIAuthServiceMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """"""
        "Пропускаем запросы из миддлвари с иными протоколами"
        if scope['type'] not in {'http', }:
            await self.app(scope, receive, send)
            return

        # Получаем client host из scope
        # scope["client"] = (host, port) или None
        client = scope.get("client")
        client_host = client[0] if client else ''
        
       
        if client_host in env.trusted_proxies:
            await self.app(scope, receive, send)
            return

        "Отклоняем, если нет заголовка от другого сервиса"
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"x-auth-service")
        
        if not auth_header or not secrets.compare_digest(
            # auth_header.decode(), env.auth_service_secret Будем переходить на mTLS с аутентификацией по сертификатам
            auth_header.decode(), env.auth_service_secret
        ):
            response = JSONResponse(
                status_code=401,
                content={"detail": "Доступ запрещён"}
            )
            await response(scope, receive, send)
            return

        "Обрабатываем запрос дальше"
        await self.app(scope, receive, send)



class ASGILoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        method, path = scope.get("method", ""), scope.get("raw_path", b"").decode("utf-8")
        start = time.perf_counter()
        status_code = 500  # По умолчанию, если что-то пойдет не так

        async def send_wrapper(message):
            nonlocal status_code
            if message['type'] == 'http.response.start':
                status_code = message['status']
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start

            "Логируем для мониторинга"
            if env.app_mode != 'local' and path != '/api/v1/healthcheck':
                log_event(f'HTTP \033[33m{method}\033[0m {path}', http_status=status_code, response_time=round(duration, 4))
            if duration > 7.0:
                log_event(f'Долгий ответ | {duration: .4f}', level='WARNING')