from pathlib import Path

import torch
from PIL import Image

from ml.config import env
from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset
from ml.detector.models import WordDetector
from ml.logger_config import log_event
from ml.word_decoder.dataset_class.dataclass_word_decoder import CRNNWordDataset
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
        else:
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
        # Сохраняем оригинальный размер
        orig_width, orig_height = img.size
        
        # Конвертируем в grayscale если нужно
        if img.mode != 'L':
            img_gray = img.convert('L')
        else:
            img_gray = img
        
        # Преобразуем в тензор (ресайз до 1280x1280)
        img_tensor = self.detector.dataset_obj.transform(img_gray).unsqueeze(0).to(env.device)
        
        # Получаем размер после трансформации (должен быть 1280x1280)
        resized_size = self.detector.img_size
        
        # Forward через детектор
        self.detector.model.eval()
        with torch.no_grad():
            preds = self.detector.model(img_tensor)
        
        # NMS (Non-Maximum Suppression)
        from ultralytics.utils.nms import non_max_suppression
        
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
        bboxes_resized = detections[:, :4]  # [x1, y1, x2, y2]
        
        # Масштабируем bboxes обратно к оригинальному размеру
        scale_x = orig_width / resized_size
        scale_y = orig_height / resized_size
        
        bboxes_original = []
        for bbox in bboxes_resized:
            x1, y1, x2, y2 = bbox
            x1_orig = x1 * scale_x
            y1_orig = y1 * scale_y
            x2_orig = x2 * scale_x
            y2_orig = y2 * scale_y
            bboxes_original.append([x1_orig, y1_orig, x2_orig, y2_orig])
        
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
        img_tensor = self.word_decoder.dataset_obj.transform(word_img).unsqueeze(0).to(env.device)
        
        # Forward через CRNN
        self.word_decoder.model.eval()
        with torch.no_grad():
            log_probs = self.word_decoder.model(img_tensor)  # [seq_len, batch, num_classes]
        
        # CTC декодирование
        from ml.word_decoder.metrics import decode_predictions
        
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
        if len(words_data) == 0:
            return []
        
        # Добавляем центры bbox для сортировки
        for word in words_data:
            x1, y1, x2, y2 = word["bbox"]
            word["center_x"] = (x1 + x2) / 2
            word["center_y"] = (y1 + y2) / 2
            word["height"] = y2 - y1
        
        # Группируем по строкам (tolerance = средняя высота слова)
        avg_height = sum(w["height"] for w in words_data) / len(words_data)
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
        if len(words_data) == 0:
            return ""
        
        # Группируем по строкам (уже отсортировано)
        avg_height = sum(w["height"] for w in words_data) / len(words_data)
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