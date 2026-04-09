import os.path
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from ultralytics.utils.ops import xywh2xyxy

from ml.base_utils import visualize_bboxes
from ml.config import WORKDIR
from ml.detector.dataset_class.dataclass_detector import DetectorAugment
from ml.logger_config import log_event


class HWR200DetectorDataset(Dataset):
    def __init__(self, path, mode: Literal['val', 'train'],transform = None, img_size: int = 1280, auto_load: bool = True, raise_mismatch_samples: bool = True):
        self.path = Path(path)
        self.transform = DetectorAugment(mode, img_size=img_size) if transform is None else transform
        self.raise_if_unexists_annotations = raise_mismatch_samples

        self.classes = ['word']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}

        self.img_extensions = {'.jpg', '.jpeg', '.png'}
        self.data: list[dict] = []
        if auto_load:
            self.create_data()

    def create_data(self):
        """
        CVAT разметка ultralytics yolo detection 1.0 format автоматически формирует train.txt с относительными путями до изображений
        Можно упростить create_data. Можно читать train.txt и подменять image-семплы на .txt аннотации(один цикл вместо 4х)

        Или таргеты в формате CVAT Images 1.1. с абсолютными координатами
        """
        target_data = []
        skipped_samples = []

        for author in self.path.iterdir():
            "Уровень директорий (0, 1, 2, ..., 199)"
            if not author.is_dir():
                continue

            for sample_dir in author.iterdir():
                "Уровень директорий (original1, reuse11, ..., fpr12)"

                "Скип, если текстовая аннотация"
                if not sample_dir.is_dir():
                    continue

                for photo_shooting_type_dir in sample_dir.iterdir():
                    "Уровень директорий (Сканы, ФотоСветлое, ФотоТемное)"

                    "Скип, если текстовая аннотация"
                    if not photo_shooting_type_dir.is_dir():
                        continue

                    for img_sample in photo_shooting_type_dir.iterdir():
                        "Уровень директорий (1.JPG, 2.JPEG, 3.PNG, ..., 1.txt, 2.txt, 3.txt)"

                        if img_sample.suffix.lower() not in self.img_extensions:
                            continue

                        "Ищем аннотацию с координатами GT боксов"
                        bboxes_path = Path(f"{os.path.splitext(img_sample)[0]}{'.txt'}")

                        if not bboxes_path.exists():
                            skipped_samples.append(img_sample)
                            continue


                        "Собираем словарь GT боксов"
                        gt_boxes = []
                        with open(str(bboxes_path), 'rt') as f:
                            for line in f.readlines():
                                "Забираем координаты. Пример ббокса в файле: "
                                # 0 0.172077 0.888471 0.118656 0.033990
                                # cls xcenter ycenter  width    height
                                # класс не нужен, т.к. он всего один
                                gt_boxes.append(list(map(float, line.split()[1:])))

                        target_data.append({
                            'photo_path': img_sample,
                            'gt_boxes': gt_boxes,
                            'labels': [self.class_to_idx['word']] * len(gt_boxes),
                        })

        skip_len = len(skipped_samples)
        if skip_len > 0:
            log_event(f'Не найдены аннотации на \033[31m{skip_len}\033[0m | {skipped_samples}', level='CRITICAL')

            if self.raise_if_unexists_annotations:
                raise RuntimeError('Не все изображения имеют текстовые аннотации')

        self.data = target_data
        return self

    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int):
        sample = self.data[idx]

        "Загружаем изображение напрямую"
        path_img = sample['photo_path']
        img = Image.open(path_img)

        "Формируем метки и GTB"
        W, H = img.size
        boxes = sample['gt_boxes']  # Нормализованные XYWH (0-1)
        labels = sample['labels']

        # Денормализуем координаты: нормализованные (0-1) -> абсолютные пиксели
        boxes_np = np.array(boxes, dtype=np.float32)  # [N, 4] XYWH
        scale = np.array([W, H, W, H], dtype=np.float32)
        boxes_xywh = boxes_np * scale  # [N, 4] XYWH в пикселях
        
        # Конвертируем XYWH -> XYXY для torchvision v2 трансформаций
        boxes_xyxy = np.zeros_like(boxes_xywh)
        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2 = cy + h/2
        
        boxes = tv_tensors.BoundingBoxes(
            boxes_xyxy,
            format="XYXY",  # Теперь в XYXY для правильной работы трансформаций
            canvas_size=(H, W)
        )
        target = {
            "boxes": boxes.to(torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        "Применяем трансформации"
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        """
        Collate function для HWR200 датасета.
        
        ВАЖНО: После torchvision v2 трансформаций bbox в формате XYXY.
        Конвертируем XYXY -> XYWH (center format) для YOLO.
        """
        images = []
        all_boxes = []
        all_batch_idx = []

        for i, (img, target) in enumerate(batch):
            images.append(img)

            boxes = target['boxes']  # В формате XYXY после трансформаций
            if boxes.numel() > 0:
                num_boxes = boxes.shape[0]
                all_boxes.append(boxes)
                all_batch_idx.append(torch.full((num_boxes,), i, dtype=torch.long))
        
        # Стекаем изображения
        images = torch.stack(images, dim=0)  # [B, 1, H, W]
        
        # Векторизованная конвертация XYXY -> XYWH + нормализация
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)  # [N, 4] XYXY в пикселях
            all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [N]
            
            h, w = images.shape[2:]
            
            # XYXY -> XYWH (center format)
            box_w = all_boxes[:, 2] - all_boxes[:, 0]  # width
            box_h = all_boxes[:, 3] - all_boxes[:, 1]  # height
            cx = all_boxes[:, 0] + box_w * 0.5  # center_x
            cy = all_boxes[:, 1] + box_h * 0.5  # center_y
            
            # Нормализация
            normalized_boxes = torch.stack([
                cx / w,
                cy / h,
                box_w / w,
                box_h / h
            ], dim=1)  # [N, 4]
            
            # Формируем финальный тензор [N, 6]: [batch_idx, cls, cx, cy, w, h]
            cls_idx = torch.zeros((normalized_boxes.shape[0], 1), dtype=torch.float32)
            targets = torch.cat([
                all_batch_idx.unsqueeze(1).float(),
                cls_idx,
                normalized_boxes
            ], dim=1)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)
        
        return images, targets

"Проверка семпла от подачи на вход, до подачи в модель"
"после абсолютизации координат перед boundingBoxes, трансформаций, нормализации координат в collate_fn)"
# dset = HWR200DetectorDataset(WORKDIR / 'dataset/HWR200/mobile_net_crops/core', 'val', img_size=1280)
# # Проверяем формат bbox ДО collate_fn
# sample0 = dset[0]
# img0, target0 = sample0
# log_event(f"До collate: boxes format = {target0['boxes'].format}", level='CRITICAL')
# log_event(f"До collate: boxes shape = {target0['boxes'].shape}", level='CRITICAL')
# log_event(f"До collate: первые 3 bbox =\n{target0['boxes'][:3]}", level='CRITICAL')
#
# collated = HWR200DetectorDataset.collate_fn([dset[0], dset[1]])
# images, targets = collated
#
# log_event(f"Images shape: {images.shape}")  # [B, 1, H, W]
# log_event(f"Targets shape: {targets.shape}")  # [N, 6] где N = сумма всех bbox
# log_event(f"Первые 5 targets:\n{targets[:5]}")  # [batch_idx, cls, cx, cy, w, h]
#
# # Визуализируем первое изображение из батча
# batch_idx = 0
# img = images[batch_idx]  # [1, H, W]
# h, w = img.shape[1], img.shape[2]
#
# # Извлекаем bbox для этого изображения
# mask = targets[:, 0] == batch_idx  # Фильтруем по batch_idx
# img_targets = targets[mask]  # [N_img, 6]
#
# log_event(f"Bbox для изображения {batch_idx}: {img_targets.shape[0]} штук")
#
# if img_targets.shape[0] > 0:
#     # Извлекаем нормализованные XYWH координаты (столбцы 2-5)
#     boxes_norm_xywh = img_targets[:, 2:6]  # [N, 4] - [cx, cy, w, h] нормализованные
#
#     # Денормализуем: (0-1) -> пиксели
#     boxes_xywh = boxes_norm_xywh.clone()
#     boxes_xywh[:, 0] *= w  # cx
#     boxes_xywh[:, 1] *= h  # cy
#     boxes_xywh[:, 2] *= w  # width
#     boxes_xywh[:, 3] *= h  # height
#
#     # Конвертируем XYWH -> XYXY
#     boxes_xyxy = xywh2xyxy(boxes_xywh)
#
#     log_event(f"Первые 3 bbox (XYXY, пиксели):\n{boxes_xyxy[:3]}")
#
#     visualize_bboxes(img, boxes_xyxy, show=True)
# else:
#     log_event("Нет bbox для визуализации!", level='ERROR')

"Проверка семпла после трансформаций"
# dset = HWR200DetectorDataset(WORKDIR / 'dataset/HWR200/25_samples/train', 'train', img_size=1280)
# sample, target = dset[1]
#
# log_event(len(target['boxes']), level='INFO')
#
# visualize_bboxes(sample, target['boxes'], show=True)

# dset = HWR200DetectorDataset(WORKDIR / 'dataset/HWR200/160_trainval/train', 'train', img_size=1280, raise_mismatch_samples=False)
# log_event(len(dset))