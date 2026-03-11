import time

from starlette.requests import Request
from starlette.types import Scope, Receive, Send, ASGIApp

from web.config import env
from web.utils.anything import get_client_ip
from web.utils.logger_config import log_event


class ASGILoggingMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope['type'] not in {'http', 'websocket'}:
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)

        ip = get_client_ip(request)
        request.state.client_ip = ip

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
            if env.app_mode != 'local' and request.url.path != '/api/v1/healthcheck':
                log_event(f'HTTP \033[33m{request.method}\033[0m {request.url.path}', request=request,
                          http_status=status_code, response_time=round(duration, 4))
            if duration > 7.0:
                log_event(f'Долгий ответ | {duration: .4f}', request=request, level='WARNING')
