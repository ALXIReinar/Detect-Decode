from dataclasses import dataclass

from starlette.requests import Request

from web.config import env

@dataclass
class ImgStatuses:
    processing: int = 1     # "В обработке"
    success: int = 2        # "Успешно"


def get_client_ip(request: Request):
    xff = request.headers.get('X-Forwarded-For')
    ip = xff.split(',')[0].strip() if (
            xff and request.client.host in env.trusted_proxies
    ) else request.client.host
    return ip
