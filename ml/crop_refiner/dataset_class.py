import os
from pathlib import Path

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from ml.config import WORKDIR
from ml.logger_config import log_event


class CropRefinerDataset(Dataset):
    def __init__(self, data_dir: Path, target_size=(128, 512), is_train=True, auto_load=True):
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
        target_h, target_w = 128, 512

        transforms = []
        if self.is_train:
            transforms.extend([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ])

        # Просто растягиваем/сжимаем до нужного прямоугольника.
        # Albumentations сам ИДЕАЛЬНО пересчитает координаты боксов.
        transforms.extend([
            A.Resize(height=target_h, width=target_w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

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
                        core_bboxes_path = core_bounding_path / Path(f"{os.path.splitext(img_extent_path.relative_to(extent_bounding_path))[0]}{'.txt'}")
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
        debug_samples = 0  # Для отладки
        
        for sample in tqdm(target_data, desc='Cropping'):

            skipped_crops_sample = 0

            extent_img = Image.open(sample['img_extent_path'])
            extent_img_extension = sample['img_extent_path'].suffix.lower()
            core_img = Image.open(sample['img_core_path'])

            main_e_w, main_e_h = extent_img.size
            main_c_w, main_c_h = core_img.size
            
            # Отладка первых нескольких сэмплов
            for extent_box in sample['gt_extent_boxes']:
                e_cx, e_cy, e_w_rel, e_h_rel = extent_box

                # Приводим в абсолютные координаты на EXTENT изображении
                e_cx_abs = e_cx * main_e_w
                e_cy_abs = e_cy * main_e_h
                e_w_abs = e_w_rel * main_e_w
                e_h_abs = e_h_rel * main_e_h
                
                # Границы Extent бокса
                e_x_min = e_cx_abs - e_w_abs / 2
                e_y_min = e_cy_abs - e_h_abs / 2
                e_x_max = e_cx_abs + e_w_abs / 2
                e_y_max = e_cy_abs + e_h_abs / 2
                
                # Ищем соответствующий Core бокс (тот, который внутри Extent бокса)
                matching_core_box = None
                max_iou = 0.0
                
                for core_box in sample['gt_core_boxes']:
                    c_cx, c_cy, c_w_rel, c_h_rel = core_box
                    
                    # Приводим Core бокс в абсолютные координаты
                    c_cx_abs = c_cx * main_e_w
                    c_cy_abs = c_cy * main_e_h
                    c_w_abs = c_w_rel * main_e_w
                    c_h_abs = c_h_rel * main_e_h
                    
                    # Проверяем, что центр Core бокса внутри Extent бокса
                    if e_x_min <= c_cx_abs <= e_x_max and e_y_min <= c_cy_abs <= e_y_max:
                        # Вычисляем IoU для выбора лучшего совпадения
                        c_x_min = c_cx_abs - c_w_abs / 2
                        c_y_min = c_cy_abs - c_h_abs / 2
                        c_x_max = c_cx_abs + c_w_abs / 2
                        c_y_max = c_cy_abs + c_h_abs / 2
                        
                        # Intersection
                        inter_x_min = max(e_x_min, c_x_min)
                        inter_y_min = max(e_y_min, c_y_min)
                        inter_x_max = min(e_x_max, c_x_max)
                        inter_y_max = min(e_y_max, c_y_max)
                        
                        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
                        extent_area = e_w_abs * e_h_abs
                        core_area = c_w_abs * c_h_abs
                        union_area = extent_area + core_area - inter_area
                        
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > max_iou:
                            max_iou = iou
                            matching_core_box = (c_cx_abs, c_cy_abs, c_w_abs, c_h_abs)
                
                # Если не нашли соответствующий Core бокс, пропускаем
                if matching_core_box is None:
                    if debug_samples <= 3:
                        log_event(f"  SKIP: Не найден Core бокс для Extent=[{e_cx_abs:.1f}, {e_cy_abs:.1f}, {e_w_abs:.1f}, {e_h_abs:.1f}]", level='WARNING')
                    skipped_crops_sample += 1
                    continue
                
                c_cx_abs, c_cy_abs, c_w_abs, c_h_abs = matching_core_box
                
                if debug_samples <= 3 and crop_id < 3:
                    log_event(f"  Matched: Extent=[{e_cx_abs:.1f}, {e_cy_abs:.1f}, {e_w_abs:.1f}, {e_h_abs:.1f}]")
                    log_event(f"  Matched: Core=[{c_cx_abs:.1f}, {c_cy_abs:.1f}, {c_w_abs:.1f}, {c_h_abs:.1f}], IoU={max_iou:.3f}")
                
                # Проверяем, что Core бокс меньше Extent бокса
                if c_w_abs > e_w_abs or c_h_abs > e_h_abs:
                    if debug_samples <= 3 and crop_id < 3:
                        log_event(f"  SKIP: Core больше Extent!", level='WARNING')
                    skipped_crops_sample += 1
                    continue

                # Вырезаем Extent кроп
                e_x_min = e_cx_abs - e_w_abs / 2
                e_y_min = e_cy_abs - e_h_abs / 2
                e_x_max = e_cx_abs + e_w_abs / 2
                e_y_max = e_cy_abs + e_h_abs / 2
                
                e_crop = extent_img.crop((e_x_min, e_y_min, e_x_max, e_y_max))
                e_crop.save(output_path / f'{crop_id:06}{extent_img_extension}')

                # Пересчитываем Core бокс в координаты относительно Extent кропа
                # Центр Core бокса относительно левого верхнего угла Extent кропа
                local_c_cx = c_cx_abs - e_x_min
                local_c_cy = c_cy_abs - e_y_min

                # Нормализуем относительно размеров Extent кропа
                norm_cx = local_c_cx / e_w_abs
                norm_cy = local_c_cy / e_h_abs
                norm_w = c_w_abs / e_w_abs
                norm_h = c_h_abs / e_h_abs
                
                # Клиппинг координат в диапазон [0, 1] (убираем погрешности вычислений)
                norm_cx = max(0.0, min(1.0, norm_cx))
                norm_cy = max(0.0, min(1.0, norm_cy))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                # Проверяем валидность бокса (центр должен быть внутри, размеры > 0)
                # Также проверяем, что бокс не выходит за границы (с эпсилоном для погрешностей float)
                eps = 1e-2
                x_min = norm_cx - norm_w / 2
                y_min = norm_cy - norm_h / 2
                x_max = norm_cx + norm_w / 2
                y_max = norm_cy + norm_h / 2
                
                # Клиппим границы бокса с учётом эпсилона
                x_min = max(0.0, min(1.0, x_min))
                y_min = max(0.0, min(1.0, y_min))
                x_max = max(0.0, min(1.0, x_max))
                y_max = max(0.0, min(1.0, y_max))
                
                # Проверяем, что бокс не вырожденный (имеет ненулевую площадь)
                if (x_max - x_min) <= eps or (y_max - y_min) <= eps:
                    continue
                
                # Пересчитываем нормализованные координаты из клиппнутых границ
                norm_cx = (x_min + x_max) / 2
                norm_cy = (y_min + y_max) / 2
                norm_w = x_max - x_min
                norm_h = y_max - y_min
                
                if norm_w <= 0.01 or norm_h <= 0.01:  # Слишком маленький бокс
                    continue
                
                # РАДИКАЛЬНАЯ ПРОВЕРКА: пропускаем боксы слишком близко к границам
                # Это убирает проблемы с погрешностями float
                margin = 0.001  # 0.1% отступ от границ
                final_x_min = norm_cx - norm_w / 2
                final_y_min = norm_cy - norm_h / 2
                final_x_max = norm_cx + norm_w / 2
                final_y_max = norm_cy + norm_h / 2
                
                if (final_x_min < margin or final_y_min < margin or 
                    final_x_max > (1.0 - margin) or final_y_max > (1.0 - margin)):
                    continue  # Пропускаем боксы у самых границ

                # Сохраняем в формате YOLO (cx, cy, w, h)
                # Округляем до 6 знаков после запятой и клиппим в строгий диапазон [0, 1]
                norm_cx = round(max(0.0, min(1.0, norm_cx)), 6)
                norm_cy = round(max(0.0, min(1.0, norm_cy)), 6)
                norm_w = round(max(0.0, min(1.0, norm_w)), 6)
                norm_h = round(max(0.0, min(1.0, norm_h)), 6)
                
                # Финальная проверка границ (после округления могут быть артефакты)
                x_min_final = norm_cx - norm_w / 2
                y_min_final = norm_cy - norm_h / 2
                x_max_final = norm_cx + norm_w / 2
                y_max_final = norm_cy + norm_h / 2
                
                # Если после округления вылезли за границы - клиппим ещё раз
                if x_min_final < 0 or y_min_final < 0 or x_max_final > 1 or y_max_final > 1:
                    x_min_final = max(0.0, x_min_final)
                    y_min_final = max(0.0, y_min_final)
                    x_max_final = min(1.0, x_max_final)
                    y_max_final = min(1.0, y_max_final)
                    
                    # Пересчитываем из клиппнутых границ
                    norm_cx = round((x_min_final + x_max_final) / 2, 6)
                    norm_cy = round((y_min_final + y_max_final) / 2, 6)
                    norm_w = round(x_max_final - x_min_final, 6)
                    norm_h = round(y_max_final - y_min_final, 6)
                
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