from enum import Enum

class AppMode(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    PROD = "prod"


APP_MODE_CONFIG = {
    AppMode.LOCAL: {
        "api_server_url": "api_server_url",
    },
    AppMode.DOCKER: {
        "api_server_url": "api_server_url_docker",
    },
    AppMode.PROD: {
        "api_server_url": "api_server_url_docker",
    },
}