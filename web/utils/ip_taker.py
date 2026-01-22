from fastapi import Request
from core.config_dir.config import trusted_proxies

def get_client_ip(request: Request):
    "Доверяем заголовку от клиента, в тестах маст-хев"  # proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    xff = request.headers.get('X-Forwarded-For')
    ip = xff.split(',')[0].strip() if (
            xff and request.client.host in trusted_proxies
    ) else request.client.host
    return ip
