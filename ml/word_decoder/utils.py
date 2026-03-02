import argparse

import torch
from torch import nn

from ml.base_utils import plot_curves


def plot_lr_chronology(history_arg, save_path=None, show=False):
    """
    График Learning Rate во время обучения
    """
    curves = [
        {
            "label": "Learning Rate",
            "values": history_arg["lr"],
            "color": "#E67E22"  # Оранжевый
        }
    ]


    plot_curves(
        curves=curves,
        title="LR Chronology",
        ylabel="LR Value",
        save_path=save_path,
        show=show,
    )

def plot_loss_dynamics(history_arg, save_path=None, show=False):
    """График общих метрик валидации: total loss и mAP."""
    values_list = history_arg["general_metrics"]

    curves_total_losses = [
        {
            "label": "Total Val Loss",
            "values": values_list['val_loss_list'],
            "color": "#E74C3C"  # Красный
        },
        {
            "label": "Total Train Loss",
            "values": values_list['train_loss_list'],
            "color": "#2ECC71"  # Зеленый
        },
    ]

    plot_curves(
        curves=curves_total_losses,
        title="Validation Dynamics",
        ylabel="Loss",
        save_path=save_path,
        show=show
    )


def plot_metrics_dynamics(history_arg, save_path=None, show=False):
    """График метрик при валидации: val CER, WER, ACC"""
    values_list = history_arg["general_metrics"]

    curves = [
        {
            "label": "CER",
            "values": values_list["cer_list"],
            "color": "#2ECC71"  # Зеленый
        },
        {
            "label": "WER",
            "values": values_list["wer_list"],
            "color": "#3498DB"  # Синий
        },
        {
            "label": "ACC",
            "values": values_list["acc_list"],
            "color": "#FFA500"  # Оранжевый
        }
    ]

    plot_curves(
        curves=curves,
        title="Metrics Validation Dynamics",
        ylabel="CER & WER & ACC",
        save_path=save_path,
        show=show,
    )


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
        '--img-height', '-i',
        type=int,
        default=64,
        help='Высота входного изображения (32, 64)'
    )
    parser.add_argument(
        '--workers', '-j',
        type=int,
        default=2,
        help='Количество workers для DataLoader'
    )
    return parser.parse_args()


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
