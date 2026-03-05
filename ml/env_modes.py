from enum import Enum

class AppMode(str, Enum):
    LOCAL = "local"
    DOCKER = "docker"
    PROD = "prod"


APP_MODE_CONFIG = {
    AppMode.LOCAL: {
        "kafka_host": "kafka_host",
        "kafka_port": "kafka_port",
        "detector_weights_path": "detector_weights_path",
        "word_decoder_weights_path": "word_decoder_weights_path",
    },
    AppMode.DOCKER: {
        "kafka_host": "kafka_host_docker",
        "kafka_port": "kafka_port_docker",
        "detector_weights_path": "detector_weights_path_docker",
        "word_decoder_weights_path": "word_decoder_weights_path_docker",
    },
    AppMode.PROD: {
        "kafka_host": "kafka_host_docker",
        "kafka_port": "kafka_port_docker",
    },
}