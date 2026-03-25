import os.path
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors

from ml.base_utils import visualize_bboxes
from ml.config import WORKDIR
from ml.detector.dataset_class.dataclass_detector import DetectorAugment
from ml.logger_config import log_event


class HWR200DetectorDataset(Dataset):
    def __init__(self, path, mode: Literal['val', 'train'],transform = None, img_size: int = 1280, auto_load: bool = True):
        self.path = Path(path)
        self.transform = DetectorAugment(mode, img_size=img_size) if transform is None else transform

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
                        log_event(len(gt_boxes), level='CRITICAL')
                        target_data.append({
                            'photo_path': img_sample,
                            'gt_boxes': gt_boxes,
                            'labels': [self.class_to_idx['word']] * len(gt_boxes),
                        })

        skip_len = len(skipped_samples)
        if skip_len > 0:
            log_event(f'Не найдены аннотации на \033[31m{skip_len}\033[0m | {skipped_samples}', level='CRITICAL')
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
        boxes_absolute = []
        for box in boxes:
            cx_norm, cy_norm, w_norm, h_norm = box
            cx = cx_norm * W
            cy = cy_norm * H
            w = w_norm * W
            h = h_norm * H
            boxes_absolute.append([cx, cy, w, h])

        boxes = tv_tensors.BoundingBoxes(
            boxes_absolute,
            format="XYWH",
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
        ВАЖНО: После torchvision v2 трансформаций bbox всегда в формате XYXY,
        даже если изначально были созданы как XYWH. Поэтому конвертация XYXY -> XYWH необходима.
        """
        images = []
        all_boxes = []
        all_batch_idx = []

        for i, (img, target) in enumerate(batch):
            images.append(img)

            boxes = target['boxes']
            if boxes.numel() > 0:
                num_boxes = boxes.shape[0]
                all_boxes.append(boxes)
                all_batch_idx.append(torch.full((num_boxes,), i, dtype=torch.long))
        
        # Стекаем изображения
        images = torch.stack(images, dim=0)  # [B, 1, H, W]
        
        # Векторизованная конвертация всех bbox сразу
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)  # [N, 4] xyxy (после трансформаций)
            all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [N]
            
            h, w = images.shape[2:]
            
            # XYXY -> XYWH (center format, векторизованно)
            box_w = all_boxes[:, 2] - all_boxes[:, 0]
            box_h = all_boxes[:, 3] - all_boxes[:, 1]
            cx = all_boxes[:, 0] + box_w * 0.5
            cy = all_boxes[:, 1] + box_h * 0.5
            
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

dset = HWR200DetectorDataset(WORKDIR / 'dataset/HWR200/25_samples/train', 'train', img_size=1280)
sample, target = dset[1]

log_event(len(target['boxes']), level='INFO')
visualize_bboxes(sample, target['boxes'], show=True)
