from xml.etree import ElementTree as ET
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from ml.logger_config import log_event


class DetectorAugment(nn.Module):
    def __init__(self, mode: Literal['train', 'val'], img_size: int = 640):
        """
        Args:
            mode: 'train' или 'val'
            img_size: размер выходного изображения (640, 960, 1280, etc.)
        """
        super().__init__()
        self.mode = mode
        self.img_size = img_size
        
        # CPU preprocessing - только базовая подготовка
        self.cpu_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Grayscale(),
            v2.Resize((img_size, img_size)),  # Параметризованный размер
            v2.ToDtype(dtype=torch.float32, scale=True),
        ])
        
        # GPU augmentations - тяжелые операции на GPU
        self.gpu_train_transform = v2.Compose([
            v2.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            v2.RandomAutocontrast(p=0.5),
            v2.SanitizeBoundingBoxes(min_size=2),
        ])

    def forward(self, img, target=None):
        # CPU preprocessing (всегда)
        if target is not None:
            img, target = self.cpu_transform(img, target)
            # Возвращаем всегда 2 значения для совместимости
            return img, target
        else:
            img = self.cpu_transform(img)
            return img

    # def forward(self, *args, **kwargs):
    #     return self.cpu_transform(*args, **kwargs)


class OCRDetectorDataset(Dataset):
    classes = ['word']
    img_formats = {'.png', '.jpg'}

    def __init__(self, path: str | Path, transform: None | Literal['train', 'val'] = None, img_size: int = 640):
        """
        Args:
            path: путь к датасету
            transform: 'train' или 'val' для применения трансформаций
            img_size: размер выходного изображения (640, 960, 1280, etc.)
        """
        self.path = path
        self.classes = ['word']
        self.img_formats = {'.png', '.jpg'}
        self.img_size = img_size

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(self.classes)}

        self.transform = DetectorAugment(transform, img_size=img_size) if transform else None
        self.data: list[dict] = self.create_data()

    def create_data(self):
        target_data = []

        path = Path(self.path)
        path_imgs = path / 'imgs'
        path_anns = path / 'annotations'

        for im_f in path_imgs.iterdir():
            ann_f = path_anns / f'{im_f.stem}.xml'
            valid_extension = im_f.suffix in self.img_formats

            if not ann_f.exists() or not valid_extension:
                continue

            img_data = {}
            ann_ps = ET.parse(ann_f)
            root = ann_ps.getroot()

            # путь к изображению
            img_data['path_img'] = im_f
            img_data['img_name'] = im_f.name

            # размер изображения
            with Image.open(im_f) as im:
                w, h = im.size
            img_data['size_img'] = [w, h]
            img_data['depth'] = 3

            objs_ann = []

            # IAM: word лежат внутри line
            for line in root.findall('.//line'):
                for word in line.findall('word'):

                    cmps = word.findall('cmp')
                    if len(cmps) == 0:
                        continue

                    xs, ys, xe, ye = [], [], [], []
                    for cmp_ in cmps:
                        x = int(cmp_.attrib['x'])
                        y = int(cmp_.attrib['y'])
                        w_ = int(cmp_.attrib['width'])
                        h_ = int(cmp_.attrib['height'])

                        xs.append(x)
                        ys.append(y)
                        xe.append(x + w_)
                        ye.append(y + h_)

                    "lt & rb углы"
                    bbox = [
                        min(xs),  # x1
                        min(ys),  # y1
                        max(xe),  # x2
                        max(ye),  # y2
                    ]

                    # бракованные координаты
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue

                    obj_ann = {
                        'bbox': bbox,
                        'label': self.class_to_idx['word'],
                        'word': word.attrib.get('text', '')
                    }

                    objs_ann.append(obj_ann)

            if len(objs_ann) == 0:
                continue

            img_data['objs_ann'] = objs_ann
            target_data.append(img_data)

        if not target_data:
            log_event(f"В папке \033[33m{self.path}\033[0m нет размеченных изображений!!!", level='CRITICAL')
            raise ValueError('нет размеченных изображений')

        target_data_len = len(target_data)
        log_event(f"В наборе данных {target_data_len} размеченных изображений", level='WARNING')
        assert target_data_len != 0, 'В указанной директории ничего нет!'

        self.data = target_data
        return target_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        "Загружаем изображение напрямую"
        path_img = sample['path_img']
        img = Image.open(path_img).convert('L')

        "Формируем метки и GTB"
        W, H = sample['size_img']
        boxes = [obj_ann['bbox'] for obj_ann in sample['objs_ann']]
        labels = [obj_ann['label'] for obj_ann in sample['objs_ann']]
        words = [obj_ann['word'] for obj_ann in sample['objs_ann']]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(H, W)
        )
        target = {
            "boxes": boxes.to(torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "words": words,
        }

        "Применяем трансформации"
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     batch: список элементов вида (img, target)
    #            target = {'boxes': [Ni, 4], 'labels': [Ni], 'words': [...]}
    #
    #     Возвращает:
    #         imgs: [B, C, H, W]
    #         targets: list of [Mi, 4] (xyxy)
    #         words: list of слов для каждой картинки
    #     """
    #     imgs, targets, words = [], [], []
    #
    #     for img, target in batch:
    #         imgs.append(img)
    #         boxes = target["boxes"]
    #         if boxes.numel() == 0:
    #             boxes = torch.zeros((0, 4), dtype=torch.float32)
    #         targets.append(boxes)
    #         words.append(target["words"])
    #
    #     imgs = torch.stack(imgs, dim=0)  # [B, C, H, W]
    #     return imgs, targets, words

    @staticmethod
    def collate_fn(batch):
        """
        Оптимизированный collate_fn с минимальными операциями на CPU.
        Конвертация bbox в нужный формат происходит векторизованно.
        """
        images = []
        all_boxes = []
        all_batch_idx = []
        words = []
        
        for i, (img, target) in enumerate(batch):  # ← ИСПРАВЛЕНО: только 2 значения
            images.append(img)
            words.append(target['words'])
            
            boxes = target['boxes']
            if boxes.numel() > 0:
                num_boxes = boxes.shape[0]
                all_boxes.append(boxes)
                all_batch_idx.append(torch.full((num_boxes,), i, dtype=torch.long))
        
        # Стекаем изображения
        images = torch.stack(images, dim=0)  # [B, 1, H, W]
        
        # Векторизованная конвертация всех bbox сразу
        if all_boxes:
            all_boxes = torch.cat(all_boxes, dim=0)  # [N, 4] xyxy
            all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [N]
            
            h, w = images.shape[2:]
            
            # xyxy -> cxcywh (векторизованно)
            box_w = all_boxes[:, 2] - all_boxes[:, 0]
            box_h = all_boxes[:, 3] - all_boxes[:, 1]
            cx = all_boxes[:, 0] + box_w * 0.5
            cy = all_boxes[:, 1] + box_h * 0.5
            
            # Нормализация (векторизованно)
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
        
        return images, targets, words