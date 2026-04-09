import os
from pathlib import Path

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from ml.logger_config import log_event


class CropRefinerDataset(Dataset):
    def __init__(self, data_dir: Path, target_size=(128, 128), is_train=True, auto_load=True):
        self.data_dir = data_dir
        self.target_size = target_size
        self.is_train = is_train

        # Собираем пары (изображение, текстовый файл)
        self.samples = []
        if auto_load:
            self.load_data()

        self.transform = self._get_transforms()

    def load_data(self):
        samples = []
        for img_path in sorted(self.data_dir.glob("*.*")):
            if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                txt_path = img_path.with_suffix('.txt')
                if txt_path.exists():
                    samples.append((img_path, txt_path))
        self.samples = samples
        return self

    def _get_transforms(self):
        # Базовые трансформации (паддинг и ресайз)
        transforms = []

        if self.is_train:
            # Аугментации цвета/яркости (не меняют боксы)
            transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                # A.SafeRotate(limit=3, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)), # Можно добавить легкий наклон
            ])

        # Сохраняем пропорции, добиваем белым фоном до квадрата
        transforms.extend([
            A.LongestMaxSize(max_size=max(self.target_size)),
            A.PadIfNeeded(
                min_height=self.target_size[0],
                min_width=self.target_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255)  # Белый паддинг
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet
            ToTensorV2()
        ])

        # format='yolo' для нормализованных xywh
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(format='yolo', label_fields=['labels'])
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path = self.samples[idx]

        # Читаем изображение
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Читаем таргет
        with open(txt_path, 'r') as f:
            # Ожидаем: cx, cy, w, h
            bbox = list(map(float, f.read().strip().split()))

        # Albumentations требует 2D список для боксов и список меток
        bboxes = [bbox]
        labels = [0]  # У нас всегда 1 класс "word"

        # Применяем трансформации (паддинг, ресайз и пересчет координат бокса)
        transformed = self.transform(image=image, bboxes=bboxes, labels=labels)

        image_tensor = transformed['image']

        # Достаем пересчитанный бокс
        if len(transformed['bboxes']) > 0:
            target_bbox = torch.tensor(transformed['bboxes'][0], dtype=torch.float32)
        else:
            # Fallback (почти невозможен с нашими аугментациями, но для безопасности)
            target_bbox = torch.tensor([0.5, 0.5, 1.0, 1.0], dtype=torch.float32)

        return image_tensor, target_bbox




    @staticmethod
    def prepare_crops(output_path: Path, extent_bounding_path: Path, core_bounding_path: Path):
        target_data = []
        skipped_samples = []

        # 1. Пробегаем по HWR200 структуре, собираем ббоксы Core и Extent
        for author in extent_bounding_path.iterdir():
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

                    for img_extent_path in photo_shooting_type_dir.iterdir():
                        "Уровень директорий (1.JPG, 2.JPEG, 3.PNG, ..., 1.txt, 2.txt, 3.txt)"

                        if img_extent_path.suffix.lower() not in {'.jpeg', '.jpg', '.png'}:
                            continue

                        "Ищем Extent аннотацию с координатами GT боксов "
                        extent_bboxes_path = Path(f"{os.path.splitext(img_extent_path)[0]}{'.txt'}")

                        if not extent_bboxes_path.exists():
                            skipped_samples.append(img_extent_path)
                            continue

                        "Ищем Core изображение и аннотацию"
                        img_core_path = core_bounding_path / img_extent_path.relative_to(extent_bounding_path)
                        core_bboxes_path = Path(f"{os.path.splitext(img_extent_path)[0]}{'.txt'}")
                        if not img_core_path.exists() or not core_bboxes_path.exists():
                            skipped_samples.append(img_extent_path)
                            continue


                        "Собираем словарь Extent GT боксов"
                        gt_extent_boxes = []
                        with open(str(extent_bboxes_path), 'rt') as f:
                            for line in f.readlines():
                                "Забираем координаты. Пример ббокса в файле: "
                                # 0 0.172077 0.888471 0.118656 0.033990
                                # cls xcenter ycenter  width    height
                                # класс не нужен, т.к. он всего один
                                gt_extent_boxes.append(list(map(float, line.split()[1:])))

                        "Собираем словарь Core GT боксов"
                        gt_core_boxes = []
                        with open(str(core_bboxes_path), 'rt') as f:
                            for line in f.readlines():
                                "Забираем координаты. Пример ббокса в файле: "
                                # 0 0.172077 0.888471 0.118656 0.033990
                                # cls xcenter ycenter  width    height
                                # класс не нужен, т.к. он всего один
                                gt_core_boxes.append(list(map(float, line.split()[1:])))

                        len_core, len_extent = len(gt_core_boxes), len(gt_extent_boxes)
                        if len_core != len_core:
                            log_event(f'Core: \033[32m{len_core}\033[0m; Extent: \033[33m{len_extent}\033[0m | {core_bboxes_path}', level='ERROR')
                            skipped_samples.append(img_extent_path)
                            continue

                        target_data.append({
                            'img_extent_path': img_extent_path,
                            'ann_extent_path': extent_bboxes_path,
                            'gt_extent_boxes': gt_extent_boxes,

                            'img_core_path': img_core_path,
                            'ann_core_path': core_bboxes_path,
                            'gt_core_boxes': gt_core_boxes,
                        })
        log_event(f'{len(target_data)} - всего форм(детектор-семплы)')
        log_event(f'{len(skipped_samples)} - скипнутые')

        # 2. Нарезаем Extent кропы, сохраняем разрешение Core кропов для таргетов
        skipped_crops = 0
        crop_id = 0
        for sample in tqdm(target_data, desc='Cropping'):

            skipped_crops_sample = 0

            extent_img = Image.open(sample['img_extent_path'])
            extent_img_extension = sample['img_extent_path'].suffix.lower()
            core_img = Image.open(sample['img_core_path'])

            main_e_w, main_e_h = extent_img.size
            main_c_w, main_c_h = core_img.size

            for extent_box, core_box in zip(sample['gt_extent_boxes'], sample['gt_core_boxes']):
                e_cx, e_cy, e_w_rel, e_h_rel = extent_box
                c_cx, c_cy, c_w_rel, c_h_rel = core_box

                # приводим в абсолютные координаты Extent
                e_cx, e_cy, e_w, e_h = e_cx * main_e_w, e_cy * main_e_h, e_w_rel * main_e_w, e_h_rel * main_e_h
                # приводим в абсолютные координаты Core
                c_w, c_h = c_w_rel * main_c_w, c_h_rel * main_c_h
                if c_w > e_w or c_h > e_h:
                    skipped_crops_sample += 1
                    continue

                # xywh2xyxy + Extent crop
                e_crop = extent_img.crop(
                    (e_cx - e_w / 2, e_cy - e_h / 2,
                     e_cx + e_w / 2, e_cy + e_h / 2)
                )
                e_crop.save(output_path / f'{crop_id:06}{extent_img_extension}')

                # сохраняем таргет(разрешение core кропа)
                # Находим абсолютные координаты центра core-бокса на исходном фото
                c_cx, c_cy = c_cx * main_c_w, c_cy * main_c_h

                # Находим координаты левого верхнего угла extent-кропа
                e_x_min = e_cx - e_w / 2
                e_y_min = e_cy - e_h / 2

                # Вычисляем центр core-бокса ВНУТРИ нашего вырезанного кропа
                local_c_cx = c_cx - e_x_min
                local_c_cy = c_cy - e_y_min

                # Нормализуем значения (от 0 до 1) относительно размеров extent-кропа
                norm_cx = local_c_cx / e_w
                norm_cy = local_c_cy / e_h
                norm_w = c_w / e_w
                norm_h = c_h / e_h

                # Сохраняем в формате YOLO (cx, cy, w, h)
                with open(output_path / f'{crop_id:06}.txt', 'wt', encoding='utf8') as f:
                    f.write(f'{norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}')
                crop_id += 1
            skipped_crops += skipped_crops_sample
        total_crops = sum(list(map(lambda x: len(x['gt_extent_boxes']), target_data)))
        log_event(f'Пропущено кропов: \033[36m{skipped_crops}\033[0m | Всего кропов: \033[35m{total_crops - skipped_crops}\033[0m')



# CropRefinerDataset.prepare_crops(
#     WORKDIR / 'dataset/HWR200/mobile_net_crops/dataset',
#     WORKDIR / 'dataset/HWR200/mobile_net_crops/extent',
#     WORKDIR / 'dataset/HWR200/mobile_net_crops/core',
# )

