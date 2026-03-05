import argparse


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Тестирование модели детектора OCR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--weights', '-w',
        type=str,
        required=True,
        help='Путь к файлу весов модели (.pth)'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Размер батча для тестирования'
    )

    parser.add_argument(
        '--img-size', '-i',
        type=int,
        default=1280,
        help='Размер входного изображения (640, 960, 1280, etc.)'
    )

    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=2,
        help='Количество workers для DataLoader'
    )

    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='Порог confidence для NMS'
    )

    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.45,
        help='Порог IoU для NMS'
    )

    return parser.parse_args()
