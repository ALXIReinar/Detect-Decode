from enum import Enum

class AppMode(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    PROD = "prod"


APP_MODE_CONFIG = {
    AppMode.LOCAL: {
        "api_server_url": "api_server_url",
        'redis_host': 'redis_host',
        'redis_port': 'redis_port',
    },
    AppMode.DOCKER: {
        "api_server_url": "api_server_url_docker",
        'redis_host': 'redis_host_docker',
        'redis_port': 'redis_port_docker',
    },
    AppMode.PROD: {
        "api_server_url": "api_server_url_docker",
        'redis_host': 'redis_host_docker',
        'redis_port': 'redis_port_docker',
    },
}