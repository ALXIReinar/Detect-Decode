from xml.etree import ElementTree as ET
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from core.utils_general.logger_config import log_event


class DetectorAugment(nn.Module):
    def __init__(self, mode: Literal['train', 'val']):
        super().__init__()
        self.mode = mode
        inference_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=torch.uint8, scale=True),
                v2.Grayscale(),
                v2.Resize((640, 640)), # Уменьшение изображения в 4,7 раза от изначального при разрешении ~2400х3500
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(std=(0.5,), mean=(0.5,)),   # для более точных значений - https://share.google/aimode/1bhH2c9qWe2aGI8ia
            ]
        )
        train_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(dtype=torch.uint8, scale=True),
                v2.Grayscale(),
                v2.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2)
                ),
                v2.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                ),
                v2.Resize((640, 640)),
                v2.SanitizeBoundingBoxes(min_size=2),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(std=(0.5,), mean=(0.5,)),
            ]
        )
        self.transforms = {
            'train': train_transform,
            'val': inference_transform,
        }

    def forward(self, *args, **kwargs):
        return self.transforms[self.mode](*args, **kwargs)


class OCRDetectorDataset(Dataset):
    def __init__(self, path: str | Path, transform: None | Literal['train', 'val'] = None):
        self.path = path
        self.classes = ['word']

        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(self.classes)}

        self.transform = DetectorAugment(transform) if transform else None
        self.data: list[dict] = []

    def create_data(self):
        target_data = []

        path = Path(self.path)
        path_imgs = path / 'imgs'
        path_anns = path / 'annotations'

        for im_f in path_imgs.iterdir():
            ann_f = path_anns / f'{im_f.stem}.xml'
            if not ann_f.exists():
                continue

            img_data = {}
            ann_ps = ET.parse(ann_f)
            root = ann_ps.getroot()

            # путь к изображению
            img_data['path_img'] = im_f

            # кэш изображения на диске
            np_f = im_f.with_suffix('.npy')
            if not np_f.exists():
                np.save(np_f, np.array(Image.open(im_f).convert('L')))

            img_data['img_name'] = im_f.name

            # размер изображения
            with Image.open(im_f) as im:
                w, h = im.size
            img_data['size_img'] = [w, h]
            img_data['depth'] = 1

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
                        max(ye)  # y2
                    ]

                    # бракованные координаты
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue

                    obj_ann = {
                        'label': self.class_to_idx['word'],
                        'bbox': bbox,
                        'text': word.attrib.get('text', '')
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

        "Находим кэшированный образец-картинку"
        path_img = sample['path_img']
        np_f = Path(path_img).with_suffix(".npy")
        if np_f.exists():
            img = np.load(np_f)
            img = torch.as_tensor(img)
        else:
            img = Image.open(path_img).convert('L')

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
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        "Применяем трансформации"
        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    @staticmethod
    def collate_fn(batch):
        imgs, bboxes = list(zip(*batch))
        return torch.stack(imgs), bboxes
