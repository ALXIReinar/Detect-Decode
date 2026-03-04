from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2
from ultralytics.utils.nms import non_max_suppression

from ml.config import env
from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset
from ml.detector.models import WordDetector
from ml.logger_config import log_event
from ml.word_decoder.dataset_class.dataclass_word_decoder import CRNNWordDataset
from ml.word_decoder.metrics import decode_predictions
from ml.word_decoder.models import CRNNWordDecoder


class CRNNModel:
    def __init__(self, weights_path: Path):
        self.img_height: None | int = None
        self.charset = None
        
        "Загружаем модель"
        self.model = self.init_model(weights_path)

        "Проверяем после загрузки"
        if self.img_height is None:
            raise ValueError('CRNN weights must include img_height!!!')

        if self.charset is None:
            raise ValueError('CRNN weights must include charset!!!')

        "Для доступа к специфичным методам (idx2char) и трансформации"
        log_event(f'Выгрузка модели | Charset: \033[35m{len(self.charset)}\033[0m, img_height: \033[34m{self.img_height}\033[0m, Charset_slice: \033[33m{self.charset[:7]}\033[0m', level='WARNING')
        self.dataset_obj = CRNNWordDataset(
            path='', charset_path=self.charset,img_height=self.img_height,transform='test', auto_load=False
        )


    def init_model(self, weights_path: Path):
        model_hyperparams = torch.load(weights_path, map_location=env.device, weights_only=False)
        self.img_height = model_hyperparams['img_height']
        self.charset = model_hyperparams['charset']

        model_inner_params = model_hyperparams['model_params']
        hidden_size, num_lstm_layers = model_inner_params['hidden_size'], model_inner_params['num_lstm_layers']
        num_classes, lstm_dropout = model_inner_params['num_classes'], model_inner_params['lstm_dropout']

        model = CRNNWordDecoder(num_classes, hidden_size, num_lstm_layers, lstm_dropout).to(env.device)
        model.load_state_dict(model_hyperparams['state_model'])
        return model


    def __call__(self, img_path):
        x = Image.open(img_path).convert('L')

        "Добавляем bs dim и перемещаем на device"
        x = self.dataset_obj.transform(x).unsqueeze(0).to(env.device)
        x = self.model(x)
        return x


class DetectorModel:
    def __init__(self, weights_path: Path):
        self.img_size: None | int = None
        self.model = self.init_model(weights_path)

        "Проверяем после загрузки"
        if self.img_size is None:
            raise ValueError('CRNN weights must include img_height!!!')

        "Для доступа к специфичным методам (idx2cls) и трансформации"
        log_event(f'Выгрузка модели | Img_size: \033[36m{self.img_size}\033[0m', level='WARNING')

        self.dataset_obj = OCRDetectorDataset('', 'val', self.img_size, False)


    def init_model(self, weights_path: Path):
        model_hyperparams = torch.load(weights_path, map_location=env.device, weights_only=False)
        model_weights = model_hyperparams['state_model']
        self.img_size = model_hyperparams['img_size']

        model_detector = WordDetector()
        model_detector.to(env.device)
        model_detector.load_state_dict(model_weights)
        return model_detector


    def __call__(self, img_path):
        self.model.train()
        x = Image.open(img_path).convert('L')

        "Добавляем bs dim и перемещаем на device"
        x = self.dataset_obj.transform(x).unsqueeze(0).to(env.device)
        x = self.model(x)
        print(x.keys())
        print(x['boxes'].shape)
        print(x['scores'].shape)
        return x


class OCRModel:
    """
    Полный OCR пайплайн: детекция слов + распознавание текста.
    
    Pipeline:
        1. Detector → bboxes слов
        2. Crop → вырезаем слова по bboxes
        3. Word Decoder → распознаём каждое слово
        4. Sort & Merge → сортируем по координатам и склеиваем текст
    """
    
    def __init__(self, detector_weights_path: Path, word_decoder_weights_path: Path, 
                 conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 600):
        """
        Инициализация OCR модели.
        
        Args:
            detector_weights_path: путь к весам детектора
            word_decoder_weights_path: путь к весам word decoder
            conf_thres: порог уверенности для NMS (default: 0.25)
            iou_thres: порог IoU для NMS (default: 0.45)
            max_det: максимальное количество детекций (default: 600)
                    Рекомендация: 600 достаточно для IAM Forms (max 128 слов/страница)
        """
        self.detector = DetectorModel(detector_weights_path)
        self.word_decoder = CRNNModel(word_decoder_weights_path)
        
        # Параметры NMS для детектора
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward_pass(self, img: Image.Image, return_details: bool = False):
        """
        Полный OCR пайплайн.
        
        Args:
            img: PIL Image (RGB или L) - ОРИГИНАЛЬНОЕ изображение
            return_details: если True, возвращает детальную информацию (bboxes, confidences)
            
        Returns:
            если return_details=False: str (распознанный текст)
            если return_details=True: dict с полной информацией
        """
        # 1. Детекция слов (возвращает bboxes в координатах оригинального изображения)
        bboxes, confidences, class_ids = self._detect_words(img)
        
        if len(bboxes) == 0:
            return "" if not return_details else {
                "text": "",
                "words": [],
                "bboxes": [],
                "confidences": []
            }
        
        # 2. Вырезаем слова из ОРИГИНАЛЬНОГО изображения и распознаём
        words_data = []
        for bbox, conf, cls_id in zip(bboxes, confidences, class_ids):
            # Вырезаем слово из оригинального изображения
            word_img = self._crop_word(img, bbox)
            
            # Распознаём
            word_text = self._recognize_word(word_img)
            
            words_data.append({
                "text": word_text,
                "bbox": bbox,  # [x1, y1, x2, y2] в координатах оригинального изображения
                "confidence": conf,
                "class_id": cls_id
            })
        
        # 3. Сортируем слова по позиции (сверху вниз, слева направо)
        words_data = self._sort_words_by_position(words_data)
        
        # 4. Склеиваем текст
        full_text = self._merge_words_to_text(words_data)
        
        if return_details:
            return {
                "text": full_text,
                "words": [w["text"] for w in words_data],
                "bboxes": [w["bbox"] for w in words_data],
                "confidences": [w["confidence"] for w in words_data],
                "words_data": words_data  # Полная информация
            }

        return full_text
    
    def _detect_words(self, img: Image.Image):
        """
        Детекция слов на изображении.
        
        Args:
            img: оригинальное изображение (любого размера)
        
        Returns:
            bboxes: list of [x1, y1, x2, y2] в координатах ОРИГИНАЛЬНОГО изображения
            confidences: list of float
            class_ids: list of int
        """
        import numpy as np
        
        # Сохраняем оригинальный размер
        orig_width, orig_height = img.size
        
        # Преобразуем в тензор (ресайз до 1280x1280)
        img_tensor = self.detector.dataset_obj.transform(img).unsqueeze(0).to(env.device)
        
        # Получаем размер после трансформации (должен быть 1280x1280)
        resized_size = self.detector.img_size
        
        # Forward через детектор
        self.detector.model.eval()
        with torch.no_grad():
            preds = self.detector.model(img_tensor)
        
        # NMS (Non-Maximum Suppression)
        preds_nms = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=True,
            max_det=self.max_det,
            nc=self.detector.model.nc
        )
        
        # Извлекаем результаты
        if len(preds_nms) == 0 or preds_nms[0] is None or len(preds_nms[0]) == 0:
            return [], [], []
        
        detections = preds_nms[0].cpu().numpy()  # [N, 6] - [x1, y1, x2, y2, conf, cls]
        
        # Bboxes в координатах ресайзнутого изображения (1280x1280)
        bboxes_resized = detections[:, :4]  # [N, 4] - [x1, y1, x2, y2]
        
        # ВЕКТОРИЗАЦИЯ: Масштабируем все bboxes одновременно через NumPy broadcasting
        scale = np.array([orig_width / resized_size, orig_height / resized_size, 
                         orig_width / resized_size, orig_height / resized_size])
        bboxes_original = (bboxes_resized * scale).tolist()
        
        confidences = detections[:, 4].tolist()
        class_ids = detections[:, 5].astype(int).tolist()
        
        return bboxes_original, confidences, class_ids
    
    def _crop_word(self, img: Image.Image, bbox: list[float]) -> Image.Image:
        """
        Вырезает слово из изображения по bbox.
        
        Args:
            img: исходное изображение
            bbox: [x1, y1, x2, y2]
            
        Returns:
            cropped image
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Проверяем границы
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)
        
        # Вырезаем
        word_img = img.crop((x1, y1, x2, y2))
        
        return word_img
    
    def _recognize_word(self, word_img: Image.Image) -> str:
        """
        Распознаёт текст на вырезанном слове.
        
        Args:
            word_img: изображение слова
            
        Returns:
            распознанный текст
        """
        # Конвертируем в grayscale
        if word_img.mode != 'L':
            word_img = word_img.convert('L')
        
        # Преобразуем в тензор
        img_tensor = self.word_decoder.dataset_obj.transform(word_img).unsqueeze(0)
        
        # Применяем ImageNet нормализацию (как в collate_fn)
        normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        img_tensor = normalize(img_tensor)
        
        # Перемещаем на device
        img_tensor = img_tensor.to(env.device)
        
        # Forward через CRNN
        self.word_decoder.model.eval()
        with torch.no_grad():
            log_probs = self.word_decoder.model(img_tensor)  # [seq_len, batch, num_classes]
        
        # CTC декодирование

        predictions = decode_predictions(log_probs, self.word_decoder.dataset_obj, blank_idx=0)
        
        if len(predictions) > 0:
            return predictions[0]
        else:
            return ""
    
    def _sort_words_by_position(self, words_data: list[dict]) -> list[dict]:
        """
        Сортирует слова по позиции: сверху вниз, слева направо.
        
        Логика:
            1. Группируем слова по строкам (по Y координате)
            2. Внутри каждой строки сортируем по X координате
        
        Args:
            words_data: список словарей с ключами "text", "bbox", "confidence"
            
        Returns:
            отсортированный список
        """
        import numpy as np
        
        if len(words_data) == 0:
            return []
        
        # ВЕКТОРИЗАЦИЯ: Вычисляем центры и высоты для всех bbox одновременно
        bboxes = np.array([w["bbox"] for w in words_data])  # [N, 4]
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2  # [N]
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2  # [N]
        heights = bboxes[:, 3] - bboxes[:, 1]  # [N]
        
        # Добавляем вычисленные значения в словари
        for i, word in enumerate(words_data):
            word["center_x"] = centers_x[i]
            word["center_y"] = centers_y[i]
            word["height"] = heights[i]
        
        # Группируем по строкам (tolerance = средняя высота слова)
        avg_height = heights.mean()
        line_tolerance = avg_height * 0.5  # Слова в одной строке если разница по Y < 50% высоты
        
        # Сортируем по Y (сверху вниз)
        words_sorted_by_y = sorted(words_data, key=lambda w: w["center_y"])
        
        # Группируем в строки
        lines = []
        current_line = [words_sorted_by_y[0]]
        
        for word in words_sorted_by_y[1:]:
            # Если слово на той же строке (разница по Y небольшая)
            if abs(word["center_y"] - current_line[-1]["center_y"]) < line_tolerance:
                current_line.append(word)
            else:
                # Новая строка
                lines.append(current_line)
                current_line = [word]
        
        # Добавляем последнюю строку
        lines.append(current_line)
        
        # Сортируем слова внутри каждой строки по X (слева направо)
        sorted_words = []
        for line in lines:
            line_sorted = sorted(line, key=lambda w: w["center_x"])
            sorted_words.extend(line_sorted)
        
        return sorted_words
    
    def _merge_words_to_text(self, words_data: list[dict]) -> str:
        """
        Склеивает слова в текст с учётом строк.
        
        Логика:
            - Слова в одной строке разделяются пробелом
            - Строки разделяются переносом строки
        
        Args:
            words_data: отсортированный список слов
            
        Returns:
            полный текст
        """
        import numpy as np
        
        if len(words_data) == 0:
            return ""
        
        # ВЕКТОРИЗАЦИЯ: Вычисляем среднюю высоту через NumPy
        heights = np.array([w["height"] for w in words_data])
        avg_height = heights.mean()
        line_tolerance = avg_height * 0.5
        
        lines = []
        current_line = [words_data[0]["text"]]
        prev_y = words_data[0]["center_y"]
        
        for word in words_data[1:]:
            # Если на той же строке
            if abs(word["center_y"] - prev_y) < line_tolerance:
                current_line.append(word["text"])
            else:
                # Новая строка
                lines.append(" ".join(current_line))
                current_line = [word["text"]]
                prev_y = word["center_y"]
        
        # Добавляем последнюю строку
        lines.append(" ".join(current_line))
        
        # Склеиваем строки
        full_text = "\n".join(lines)
        
        return full_text

    def _recognize_words_batch(self, word_imgs: list[Image.Image], batch_size: int = 32) -> list[str]:
        """
        Распознаёт батч слов одновременно (ОПТИМИЗАЦИЯ).
        
        Использует паддинг для выравнивания ширины изображений (как в collate_fn).
        
        Args:
            word_imgs: список изображений слов
            batch_size: размер батча (default: 32)
            
        Returns:
            список распознанных текстов
        """
        import torch.nn.functional as F
        
        if len(word_imgs) == 0:
            return []
        
        all_predictions = []
        
        # Обрабатываем батчами по batch_size
        for i in range(0, len(word_imgs), batch_size):
            batch = word_imgs[i:i + batch_size]
            
            # ВЕКТОРИЗАЦИЯ: Конвертируем все изображения в grayscale одним списковым включением
            batch_gray = [img.convert('L') if img.mode != 'L' else img for img in batch]
            
            # Трансформируем все изображения
            img_tensors = [self.word_decoder.dataset_obj.transform(img) for img in batch_gray]
            orig_widths = [t.shape[2] for t in img_tensors]
            
            # Находим максимальную ширину в батче
            max_width = max(orig_widths)
            
            # ВЕКТОРИЗАЦИЯ: Паддим все изображения через torch.nn.functional.pad
            padded_tensors = []
            for img_tensor in img_tensors:
                w = img_tensor.shape[2]
                if w < max_width:
                    # F.pad: (left, right, top, bottom)
                    # Паддим справа до max_width белым цветом (1.0)
                    padding = (0, max_width - w, 0, 0)
                    padded_img = F.pad(img_tensor, padding, mode='constant', value=1.0)
                    padded_tensors.append(padded_img)
                else:
                    padded_tensors.append(img_tensor)
            
            # Стакаем в батч
            batch_tensor = torch.stack(padded_tensors)  # [batch, C, H, max_width]
            
            # Применяем ImageNet нормализацию
            normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            batch_tensor = normalize(batch_tensor)
            
            # Перемещаем на device
            batch_tensor = batch_tensor.to(env.device)
            
            # Forward через CRNN
            self.word_decoder.model.eval()
            with torch.no_grad():
                log_probs = self.word_decoder.model(batch_tensor)  # [seq_len, batch, num_classes]
            
            # CTC декодирование
            predictions = decode_predictions(log_probs, self.word_decoder.dataset_obj, blank_idx=0)
            all_predictions.extend(predictions)
        
        return all_predictions
    
    def _detect_words_batch(self, imgs: list[Image.Image]) -> list[tuple]:
        """
        Детекция слов на батче изображений (ОПТИМИЗАЦИЯ).
        
        Args:
            imgs: список изображений
            
        Returns:
            список кортежей (bboxes, confidences, class_ids) для каждого изображения
        """
        import numpy as np
        
        if len(imgs) == 0:
            return []
        
        # ВЕКТОРИЗАЦИЯ: Сохраняем оригинальные размеры как numpy array
        orig_sizes = np.array([(img.size[0], img.size[1]) for img in imgs])  # [N, 2]
        
        # Трансформируем все изображения
        img_tensors = [self.detector.dataset_obj.transform(img) for img in imgs]
        
        # Стакаем в батч
        batch_tensor = torch.stack(img_tensors).to(env.device)  # [batch, C, H, W]
        
        # Forward pass
        self.detector.model.eval()
        with torch.no_grad():
            preds = self.detector.model(batch_tensor)  # Батч предсказаний
        
        # NMS для каждого изображения в батче
        preds_nms = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            agnostic=True,
            max_det=self.max_det,
            nc=self.detector.model.nc
        )
        
        # Обрабатываем результаты для каждого изображения
        results = []
        resized_size = self.detector.img_size
        
        for i, (detections, (orig_w, orig_h)) in enumerate(zip(preds_nms, orig_sizes)):
            if detections is None or len(detections) == 0:
                results.append(([], [], []))
                continue
            
            detections_np = detections.cpu().numpy()
            
            # ВЕКТОРИЗАЦИЯ: Масштабируем все bboxes одновременно через NumPy broadcasting
            scale = np.array([orig_w / resized_size, orig_h / resized_size,
                             orig_w / resized_size, orig_h / resized_size])
            bboxes_original = (detections_np[:, :4] * scale).tolist()
            
            confidences = detections_np[:, 4].tolist()
            class_ids = detections_np[:, 5].astype(int).tolist()
            
            results.append((bboxes_original, confidences, class_ids))
        
        return results
    
    def forward_pass_batch(
        self, 
        imgs: list[Image.Image], 
        return_details: bool = False
    ) -> list[dict]:
        """
        Батчинг OCR для нескольких изображений (ОПТИМИЗАЦИЯ).
        
        Использует батчинг для детектора и word decoder для ускорения обработки.
        
        Args:
            imgs: список изображений
            return_details: возвращать детальную информацию
            
        Returns:
            список результатов для каждого изображения
        """
        if len(imgs) == 0:
            return []
        
        # 1. Детекция для всех изображений (батчинг детектора)
        batch_detections = self._detect_words_batch(imgs)
        
        results = []
        
        # 2. Обрабатываем каждое изображение
        for img, (bboxes, confidences, class_ids) in zip(imgs, batch_detections):
            if len(bboxes) == 0:
                # Нет слов на изображении
                empty_result = {
                    "text": "",
                    "words": [],
                    "bboxes": [],
                    "confidences": []
                }
                results.append(empty_result)
                continue
            
            # Вырезаем все слова
            word_imgs = [self._crop_word(img, bbox) for bbox in bboxes]
            
            # Распознаём батчем (батчинг word decoder)
            word_texts = self._recognize_words_batch(word_imgs)
            
            # Формируем words_data
            words_data = [
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id
                }
                for text, bbox, conf, cls_id in zip(word_texts, bboxes, confidences, class_ids)
            ]
            
            # Сортируем и склеиваем
            words_data = self._sort_words_by_position(words_data)
            full_text = self._merge_words_to_text(words_data)
            
            if return_details:
                results.append({
                    "text": full_text,
                    "words": [w["text"] for w in words_data],
                    "bboxes": [w["bbox"] for w in words_data],
                    "confidences": [w["confidence"] for w in words_data],
                    "words_data": words_data
                })
            else:
                results.append({"text": full_text})
        
        return results
