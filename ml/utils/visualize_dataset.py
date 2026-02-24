"""
Утилита для визуализации датасета и проверки размеров bbox после трансформаций.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from ml.dataclass_detector import OCRDetectorDataset
from ml.config import WORKDIR


def visualize_sample(dataset, idx, save_path=None):
    """
    Визуализирует один образец из датасета с bbox.
    
    Args:
        dataset: OCRDetectorDataset
        idx: индекс образца
        save_path: путь для сохранения (опционально)
    """
    img, target, _ = dataset[idx]
    
    # Конвертируем в numpy для отображения
    if isinstance(img, torch.Tensor):
        img_np = img.squeeze().cpu().numpy()  # [H, W]
    else:
        img_np = img
    
    boxes = target['boxes'].cpu().numpy()  # [N, 4] xyxy
    words = target['words']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_np, cmap='gray')
    
    h, w = img_np.shape
    
    # Статистика размеров bbox
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    
    print(f"\n{'='*60}")
    print(f"Образец #{idx}")
    print(f"Размер изображения: {w}×{h}")
    print(f"Количество слов: {len(boxes)}")
    print(f"\nСтатистика размеров bbox (в пикселях):")
    print(f"  Ширина:  min={box_widths.min():.1f}, max={box_widths.max():.1f}, mean={box_widths.mean():.1f}")
    print(f"  Высота:  min={box_heights.min():.1f}, max={box_heights.max():.1f}, mean={box_heights.mean():.1f}")
    print(f"  Площадь: min={(box_widths*box_heights).min():.1f}, max={(box_widths*box_heights).max():.1f}")
    
    # Подсчет мелких bbox (проблемных для детекции)
    small_boxes = ((box_widths < 16) | (box_heights < 16)).sum()
    tiny_boxes = ((box_widths < 8) | (box_heights < 8)).sum()
    
    print(f"\n⚠️  Мелкие bbox (<16px по любой стороне): {small_boxes} ({small_boxes/len(boxes)*100:.1f}%)")
    print(f"⚠️  Очень мелкие bbox (<8px): {tiny_boxes} ({tiny_boxes/len(boxes)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Рисуем bbox
    for i, (box, word) in enumerate(zip(boxes, words)):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Цвет в зависимости от размера
        if width < 8 or height < 8:
            color = 'red'  # Очень мелкие
            linewidth = 2
        elif width < 16 or height < 16:
            color = 'orange'  # Мелкие
            linewidth = 1.5
        else:
            color = 'green'  # Нормальные
            linewidth = 1
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Подписываем только мелкие bbox
        if width < 16 or height < 16:
            ax.text(x1, y1-2, f'{width:.0f}×{height:.0f}', 
                   fontsize=8, color=color, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_title(f'Образец #{idx} | Размер: {w}×{h} | Слов: {len(boxes)}\n'
                f'🔴 Красные: <8px | 🟠 Оранжевые: <16px | 🟢 Зеленые: ≥16px',
                fontsize=12)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return {
        'image_size': (w, h),
        'num_boxes': len(boxes),
        'box_widths': box_widths,
        'box_heights': box_heights,
        'small_boxes': small_boxes,
        'tiny_boxes': tiny_boxes
    }


def analyze_dataset_resolution(dataset, num_samples=10):
    """
    Анализирует несколько образцов и выдает рекомендации по разрешению.
    """
    all_widths = []
    all_heights = []
    all_small = 0
    all_tiny = 0
    total_boxes = 0
    
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ДАТАСЕТА ({num_samples} образцов)")
    print(f"{'='*60}\n")
    
    for i in range(min(num_samples, len(dataset))):
        img, target, _ = dataset[i]
        boxes = target['boxes'].cpu().numpy()
        
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        
        all_widths.extend(box_widths)
        all_heights.extend(box_heights)
        
        all_small += ((box_widths < 16) | (box_heights < 16)).sum()
        all_tiny += ((box_widths < 8) | (box_heights < 8)).sum()
        total_boxes += len(boxes)
    
    import numpy as np
    all_widths = np.array(all_widths)
    all_heights = np.array(all_heights)
    
    print(f"Всего bbox проанализировано: {total_boxes}")
    print(f"\nСтатистика размеров:")
    print(f"  Ширина:  min={all_widths.min():.1f}, max={all_widths.max():.1f}, mean={all_widths.mean():.1f}, median={np.median(all_widths):.1f}")
    print(f"  Высота:  min={all_heights.min():.1f}, max={all_heights.max():.1f}, mean={all_heights.mean():.1f}, median={np.median(all_heights):.1f}")
    
    print(f"\n⚠️  Проблемные bbox:")
    print(f"  Мелкие (<16px): {all_small} ({all_small/total_boxes*100:.1f}%)")
    print(f"  Очень мелкие (<8px): {all_tiny} ({all_tiny/total_boxes*100:.1f}%)")
    
    # Рекомендации
    print(f"\n{'='*60}")
    print("РЕКОМЕНДАЦИИ:")
    print(f"{'='*60}")
    
    min_size = min(all_widths.min(), all_heights.min())
    
    if all_tiny > 0:
        print(f"🔴 КРИТИЧНО: {all_tiny} bbox меньше 8px!")
        print(f"   Рекомендуемое разрешение: 1280×1280 или выше")
        recommended = 1280
    elif all_small / total_boxes > 0.3:
        print(f"🟠 ВНИМАНИЕ: {all_small/total_boxes*100:.1f}% bbox меньше 16px")
        print(f"   Рекомендуемое разрешение: 960×960 или 1280×1280")
        recommended = 960
    else:
        print(f"🟢 OK: Большинство bbox достаточно крупные")
        print(f"   Текущее разрешение 640×640 приемлемо")
        recommended = 640
    
    print(f"\nДля YOLOv8 (сетка 8×8 на мелком уровне):")
    print(f"  Минимальный размер bbox: {min_size:.1f}px")
    print(f"  Рекомендуемый минимум: 16px")
    print(f"  Оптимальное разрешение: {recommended}×{recommended}")
    print(f"{'='*60}\n")
    
    return recommended


if __name__ == '__main__':
    # Загружаем датасет
    train_dset = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'train', 'train')
    
    # Анализируем датасет
    recommended_size = analyze_dataset_resolution(train_dset, num_samples=20)
    
    # Визуализируем несколько образцов
    output_dir = WORKDIR / 'visualizations'
    output_dir.mkdir(exist_ok=True)
    
    print("\nВизуализация образцов...")
    for i in [0, 5, 10]:
        visualize_sample(train_dset, i, save_path=output_dir / f'sample_{i}.png')
