from enum import Enum

class AppMode(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    PROD = "prod"


APP_MODE_CONFIG = {
    AppMode.LOCAL: {
        "ml_server_url": "ml_server_url",
        "pg_host": "pg_host",
        "pg_port": "pg_port",
        'redis_host': 'redis_host',
        'redis_port': 'redis_port',
    },
    AppMode.DOCKER: {
        "ml_server_url": "ml_server_url_docker",
        "pg_host": "pg_host_docker",
        "pg_port": "pg_port_docker",
        'redis_host': 'redis_host_docker',
        'redis_port': 'redis_port_docker',
    },
    AppMode.PROD: {
        "ml_server_url": "ml_server_url_docker",
        "pg_host": "pg_host_docker",
        "pg_port": "pg_port_docker",
        'redis_host': 'redis_host_docker',
        'redis_port': 'redis_port_docker',
    },
}