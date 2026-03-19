import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ml.logger_config import log_event


class OCRPipelineDataset(Dataset):
    """
    Датасет для тестирования всего OCR пайплайна.
    
    Возвращает полные изображения документов с ground truth:
    - Список слов в правильном порядке (сверху вниз, слева направо)
    - Bbox для каждого слова
    - Маска поддерживаемых символов (для учёта в метриках)
    """
    
    def __init__(
        self,
        path: str | Path,
        detector_images_formats: tuple[str],
        class_to_idx: dict,
        charset: list[str] = None,
    ):
        """
        Args:
            path: путь к директории с test данными (imgs/ и annotations/)
            detector_images_formats: поддерживаемые форматы изображений
            class_to_idx: маппинг классов детектора
            charset: список поддерживаемых символов (для маски unsupported слов)
        """
        self.path = Path(path)
        log_event(self.path)
        self.detector_images_formats = detector_images_formats
        self.class_to_idx = class_to_idx
        self.charset = set(charset) if charset else None
        
        self.data = self.create_data()
        log_event(f'OCRPipelineDataset загружен | Семплов: \033[32m{len(self.data)}\033[0m', level='WARNING')
    
    def create_data(self):
        """
        Парсит аннотации и готовит данные.
        
        Для каждого семпла:
        1. Парсит XML аннотацию
        2. Извлекает слова + bbox
        3. Сортирует слова по позиции (сверху вниз, слева направо)
        4. Проверяет поддержку charset (если задан)
        """
        target_data = []
        
        path_imgs = self.path / 'imgs'
        path_anns = self.path / 'annotations'
        
        if not path_imgs.exists() or not path_anns.exists():
            raise FileNotFoundError(f"Директории imgs/ или annotations/ не найдены в {self.path}")
        
        for img_file in path_imgs.iterdir():
            ann_file = path_anns / f'{img_file.stem}.xml'
            valid_extension = img_file.suffix in self.detector_images_formats
            
            if not ann_file.exists() or not valid_extension:
                continue
            
            # Парсим XML
            tree = ET.parse(ann_file)
            root = tree.getroot()
            
            # Извлекаем слова + bbox (IAM: word лежат внутри line)
            words_data = []
            for line in root.findall('.//line'):
                for word in line.findall('word'):
                    text = word.get('text')
                    
                    # В IAM XML bbox находится в дочерних элементах <cmp>
                    cmps = word.findall('cmp')
                    if len(cmps) == 0:
                        continue
                    
                    # Вычисляем общий bbox из всех компонентов
                    x_coords = []
                    y_coords = []
                    for cmp in cmps:
                        x = int(cmp.get('x'))
                        y = int(cmp.get('y'))
                        w = int(cmp.get('width'))
                        h = int(cmp.get('height'))
                        
                        x_coords.extend([x, x + w])
                        y_coords.extend([y, y + h])
                    
                    # Общий bbox
                    x1 = min(x_coords)
                    y1 = min(y_coords)
                    x2 = max(x_coords)
                    y2 = max(y_coords)
                    
                    bbox = [x1, y1, x2, y2]
                    
                    # Проверяем поддержку charset
                    is_supported = True
                    if self.charset is not None:
                        is_supported = all(char in self.charset for char in text)
                    
                    words_data.append({
                        'text': text,
                        'bbox': bbox,
                        'is_supported': is_supported
                    })
            
            if len(words_data) == 0:
                continue
            
            # Сортируем слова по позиции (сверху вниз, слева направо)
            sorted_words = self._sort_words_by_position(words_data)
            
            target_data.append({
                'img_path': img_file,
                'words': sorted_words
            })
        
        return target_data
    
    def _sort_words_by_position(self, words_data: list[dict]) -> list[dict]:
        """
        Сортирует слова по позиции: сверху вниз, слева направо.
        
        Улучшенная логика с вертикальным перекрытием bbox:
            1. Сортируем слова по верхней границе (y_top)
            2. Группируем в строки по вертикальному перекрытию (IoU по Y)
            3. Если перекрытие > 50% высоты слова → та же строка
            4. Сортируем строки по Y, слова внутри строк по X
        """
        if len(words_data) == 0:
            return []
        
        # Подготовка данных: вычисляем координаты один раз
        for w in words_data:
            x1, y1, x2, y2 = w["bbox"]
            w["y_top"] = y1
            w["y_bottom"] = y2
            w["x_left"] = x1
            w["x_right"] = x2
            w["center_x"] = (x1 + x2) / 2
            w["center_y"] = (y1 + y2) / 2
            w["height"] = y2 - y1
        
        # Сортируем все слова по верхней границе (Y)
        words_sorted_y = sorted(words_data, key=lambda x: x["y_top"])
        
        # Группировка в строки по вертикальному перекрытию
        lines = []
        if words_sorted_y:
            current_line = [words_sorted_y[0]]
            # Интервал текущей строки по вертикали
            line_y1 = words_sorted_y[0]["y_top"]
            line_y2 = words_sorted_y[0]["y_bottom"]
            
            for i in range(1, len(words_sorted_y)):
                word = words_sorted_y[i]
                
                # Вычисляем вертикальное перекрытие интервала слова и интервала строки
                overlap_y1 = max(line_y1, word["y_top"])
                overlap_y2 = min(line_y2, word["y_bottom"])
                overlap_h = max(0, overlap_y2 - overlap_y1)
                
                # Если перекрытие > 50% высоты текущего слова → это одна строка
                if overlap_h > (word["height"] * 0.5):
                    current_line.append(word)
                    # Расширяем границы строки
                    line_y1 = min(line_y1, word["y_top"])
                    line_y2 = max(line_y2, word["y_bottom"])
                else:
                    # Новая строка
                    lines.append(current_line)
                    current_line = [word]
                    line_y1, line_y2 = word["y_top"], word["y_bottom"]
            
            lines.append(current_line)
        
        # Сортируем слова внутри строк по X (слева направо)
        sorted_words = []
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x["x_left"])
            sorted_words.extend(line_sorted)
        
        return sorted_words
    
    def __getitem__(self, idx):
        """
        Возвращает:
            - image: PIL.Image (оригинальное изображение)
            - words: list[str] (слова в правильном порядке)
            - bboxes: list[list[float]] (bbox для каждого слова в формате [x1, y1, x2, y2])
            - supported_mask: list[bool] (какие слова поддерживаются charset)
            - img_ids: list[None] (для совместимости с OCRModel.forward_pass)
        """
        sample = self.data[idx]
        
        # Загружаем изображение
        img = Image.open(sample['img_path']).convert('RGB')
        
        # Извлекаем данные
        words = [w['text'] for w in sample['words']]
        bboxes = [w['bbox'] for w in sample['words']]
        supported_mask = [w['is_supported'] for w in sample['words']]
        
        # Создаём список None той же длины что и words (для OCRModel)
        img_ids = [None] * len(words)
        
        return {
            'image': img,
            'words': words,
            'bboxes': bboxes,
            'supported_mask': supported_mask,
            'img_ids': img_ids
        }
    
    def __len__(self):
        return len(self.data)


def ocr_pipeline_collate_fn(batch):
    """
    Collate function для OCRPipelineDataset.
    
    Объединяет батч семплов в удобный формат для тестирования.
    
    Args:
        batch: список словарей из __getitem__
    
    Returns:
        dict с батчем данных:
            - images: list[PIL.Image]
            - all_words: list[list[str]] - слова для каждого изображения
            - all_bboxes: list[list[list[float]]] - bbox для каждого изображения
            - all_supported_masks: list[list[bool]] - маски для каждого изображения
            - all_img_ids: list[list[None]] - img_ids для каждого изображения
    """
    images = [item['image'] for item in batch]
    all_words = [item['words'] for item in batch]
    all_bboxes = [item['bboxes'] for item in batch]
    all_supported_masks = [item['supported_mask'] for item in batch]
    all_img_ids = [item['img_ids'] for item in batch]
    
    return {
        'images': images,
        'all_words': all_words,
        'all_bboxes': all_bboxes,
        'all_supported_masks': all_supported_masks,
        'all_img_ids': all_img_ids
    }
