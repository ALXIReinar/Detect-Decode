from pathlib import Path

import torch
from matplotlib import pyplot as plt, patches

from ml.logger_config import log_event


def visualize_bboxes(image, boxes, figcolor='red', figsize=(8, 8), linewidth=1, save_path=None, show=True):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3:
            image = image.squeeze(0)
        image = image.numpy()

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=linewidth,
            edgecolor=figcolor,
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.savefig(save_path, dpi=300)
    if show:
        plt.show()


def check_imgs_anns_equal(dataset_location: Path):
    layouts = ['train', 'val', 'test']
    for layout in layouts:
        missing_files = []
        for file in (dataset_location / layout / "annotations").iterdir():
            if not Path.exists(dataset_location / layout / "imgs" / f"{file.name.split('.')[0]}.png"):
                missing_files.append(file)
        log_event(f"\033[34m{layout=}\033[0m | \033[31m{len(missing_files)=}\033[0m, \033[36m{missing_files=}\033[0m", level='WARNING')




def plot_curves(curves, title, xlabel="Epoch", ylabel="Value", save_path=None, show=True):
    plt.figure(figsize=(10, 6))

    for curve in curves:
        plt.plot(curve["values"], label=curve["label"], color=curve.get("color", None), linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def calculate_ocr_metric(
        cer: float, wer: float, map50: float, map50_95: float, metrics_weights: dict = None
) -> float:
    """
    Вычисляет комбинированную метрику для оценки всего OCR пайплайна.
    
    Учитывает качество как детектора (mAP), так и декодера (CER/WER).
    
    Args:
        cer: Character Error Rate (0-100%), от CRNN decoder
        wer: Word Error Rate (0-100%), от CRNN decoder
        map50: mAP@0.5 (0-100%), от YOLO detector
        map50_95: mAP@0.5:0.95 (0-100%), от YOLO detector
        metrics_weights: словарь весов для компонентов (опционально)
            {
                'detector': 0.4,  # вес детектора
                'decoder': 0.6    # вес декодера
            }
    
    Returns:
        float [0, 100] - Общая оценка пайплайна OCR модели
    
    Формула:
        detector_score = (mAP50 + mAP50-95) / 2
        decoder_score = 100 - (CER + WER) / 2
        pipeline_score = detector_weight * detector_score + decoder_weight * decoder_score

    """
    if metrics_weights is None:
        metrics_weights = {
            'detector': 0.4,
            'decoder': 0.6
        }
    
    "Валидация весов"
    if abs(sum(metrics_weights.values()) - 1.0) > 0.01:
        raise ValueError(f"Сумма весов должна быть 1.0, получено: {sum(metrics_weights.values())}")
    
    "Валидация входных данных"
    for name, value in [('CER', cer), ('WER', wer), ('mAP50', map50), ('mAP50-95', map50_95)]:
        if not (0 <= value <= 100):
            raise ValueError(f"{name} должен быть в диапазоне [0, 100], получено: {value}")
    

    "Оценка детектора"
    detector_score = (map50 + map50_95) / 2
    "Оценка декодера"
    decoder_accuracy = 100 - ((cer + wer) / 2) # CER и WER - это ошибки, поэтому вычитаем из 100

    "Скейлим по значимости метрик моделей и суммируем = OCR metric"
    pipeline_score = (
            metrics_weights['detector'] * detector_score +
            metrics_weights['decoder'] * decoder_accuracy
    )

    return pipeline_score
