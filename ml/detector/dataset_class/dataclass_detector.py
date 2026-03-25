from xml.etree import ElementTree as ET
from pathlib import Path
from typing import Literal
import albumentations as A
import numpy as np

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
        
        # Albumentations: pixel-level аугментации (не меняют геометрию, bbox не нужны)
        albu_transforms = [
            A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ]
        
        if mode == 'train':
            # Добавляем train-only pixel-level аугментации
            albu_transforms.extend([
                A.GaussNoise(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            ])
        
        self.albu_transform = A.Compose(albu_transforms)
        
        # Torchvision v2: геометрические трансформации (bbox-aware)
        torch_transforms = [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Resize((img_size, img_size)),  # bbox автоматически масштабируются
        ]
        
        if mode == 'train':
            # Легкие геометрические аугментации (bbox-aware)
            torch_transforms.extend([
                v2.RandomAffine(
                    degrees=3,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                v2.SanitizeBoundingBoxes(min_size=2),
            ])
        
        torch_transforms.append(v2.ToDtype(dtype=torch.float32, scale=True))
        self.torch_transform = v2.Compose(torch_transforms)

    @staticmethod
    def extract_green_channel(img: Image.Image) -> np.ndarray:
        """Извлекает зелёный канал из RGB изображения"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img.getchannel('G'), dtype=np.uint8)

    def forward(self, img, target=None):
        # 1. Извлекаем зелёный канал (PIL -> numpy)
        img_np = self.extract_green_channel(img)
        
        # 2. Применяем Albumentations (pixel-level, bbox не нужны)
        img_np = self.albu_transform(image=img_np)['image']
        img_pil = Image.fromarray(img_np, mode='L')
        
        # 4. Применяем torchvision (геометрия + bbox)
        if target is not None:
            img_tensor, target = self.torch_transform(img_pil, target)
            return img_tensor, target
        else:
            img_tensor = self.torch_transform(img_pil)
            return img_tensor



class OCRDetectorDataset(Dataset):
    def __init__(
            self, path: str | Path, transform: None | Literal['train', 'val'] = None, img_size: int = 1280, auto_load: bool = True
    ):
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
        self.data: list[dict] = []

        "Автозагрузка данных"
        if auto_load:
            self.create_data()

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
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        "Загружаем изображение напрямую"
        path_img = sample['path_img']
        img = Image.open(path_img)

        "Формируем метки и GTB"
        W, H = sample['size_img']
        boxes = [obj_ann['bbox'] for obj_ann in sample['objs_ann']]
        labels = [obj_ann['label'] for obj_ann in sample['objs_ann']]

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
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
        Оптимизированный collate_fn с минимальными операциями на CPU.
        Конвертация bbox в нужный формат происходит векторизованно.
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
            all_boxes = torch.cat(all_boxes, dim=0)  # [N, 4] xyxy
            all_batch_idx = torch.cat(all_batch_idx, dim=0)  # [N]
            
            h, w = images.shape[2:]
            
            # xyxy -> cxcywh (векторизованно)
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


# im_path = Path(r"C:\Users\ALXI\Pictures\Screenshots")
# image = Image.open(im_path / 'Снимок экрана 2025-12-18 043709.png')
# transform = DetectorAugment('val', 1280)
# image_tsr2np = transform(image).numpy()
#
# image_tsr2np = np.transpose(image_tsr2np, (1, 2, 0))
# image_tsr2np = np.squeeze(image_tsr2np)
#
# if image_tsr2np.max() <= 1.0:
#     image_tsr2np = (image_tsr2np * 255).astype(np.uint8)
# else:
#     image_tsr2np = image_tsr2np.astype(np.uint8)
#
# Image.fromarray(image_tsr2np).save(im_path / 'green_223.JPG')