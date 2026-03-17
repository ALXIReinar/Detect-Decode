from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from ultralytics.utils.nms import non_max_suppression

from ml.api_layer.utils import save_word_crop
from ml.config import env
from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset
from ml.detector.models import WordDetector
from ml.env_modes import AppMode
from ml.logger_config import log_event
from ml.word_decoder.dataset_class.dataclass_word_decoder import CRNNWordDataset
from ml.word_decoder.metrics import decode_predictions
from ml.word_decoder.models import CRNNWordDecoder
from ml.word_decoder.dataset_class.beam_search_decoder import BeamSearchDecoder
from ml.word_decoder.spell_checker import SpellChecker


class CRNNModel:
    def __init__(self, weights_path: Path, use_beam_search: bool = False):
        """
        Args:
            weights_path: путь к весам CRNN модели
            use_beam_search: использовать beam search decoder вместо greedy (default: False)
        """
        self.img_height = None
        self.charset = None
        self.beam_size = None

        self.use_beam_search = use_beam_search
        
        "Загружаем модель"
        self.model = self.init_model(weights_path)

        "Проверяем после загрузки"
        if self.img_height is None:
            raise ValueError('CRNN weights must include img_height!!!')

        if self.charset is None:
            raise ValueError('CRNN weights must include charset!!!')

        if self.beam_size is None:
            raise ValueError('CRNN weights must include beam_size!!!')

        "Для доступа к специфичным методам (idx2char) и трансформации"
        log_event(f'Выгрузка модели | Charset: \033[35m{len(self.charset)}\033[0m, img_height: \033[34m{self.img_height}\033[0m,  Beam Size: \033[31m{self.beam_size}\033[0m Charset_slice: \033[33m{self.charset[:7]}\033[0m', level='WARNING')
        self.dataset_obj = CRNNWordDataset(
            path='', charset_path=self.charset,img_height=self.img_height,transform='test', auto_load=False
        )
        
        "Инициализация beam search decoder (опционально)"
        if self.use_beam_search:
            self.beam_search_decoder = BeamSearchDecoder(
                tokens=self.charset,
                beam_size=self.beam_size,
                nbest=1,
                use_cuda=True  # Используем CUDA если доступен
            )
            log_event(f'\033[34mBeam search decoder\033[0m | type=\033[31m{self.beam_search_decoder.decoder_type}\033[0m', level='WARNING')
        else:
            self.beam_search_decoder = None


    def init_model(self, weights_path: Path):
        model_hyperparams = torch.load(weights_path, map_location=env.device, weights_only=False)
        self.img_height = model_hyperparams['img_height']
        self.charset = model_hyperparams['charset']
        self.beam_size = model_hyperparams.get('beam_size', 10)

        model_inner_params = model_hyperparams['model_params']
        hidden_size, num_lstm_layers = model_inner_params['hidden_size'], model_inner_params['num_lstm_layers']
        num_classes, lstm_dropout = model_inner_params['num_classes'], model_inner_params['lstm_dropout']
        
        # Backward compatibility: старые веса не имеют use_feature_compressor
        use_feature_compressor = model_inner_params.get('use_feature_compressor', False)
        compressor_output_size = model_inner_params.get('compressor_output_size', 512)

        model = CRNNWordDecoder(
            num_classes, hidden_size, num_lstm_layers, lstm_dropout,
            use_feature_compressor=use_feature_compressor,
            compressor_output_size=compressor_output_size
        ).to(env.device)
        model.load_state_dict(model_hyperparams['state_model'])
        
        # Логируем архитектуру
        if use_feature_compressor:
            log_event(f'Архитектура: \033[32mOptimized\033[0m (feature_compressor: 2048 → {compressor_output_size})', level='WARNING')
        else:
            log_event(f'Архитектура: \033[33mLegacy\033[0m (BiLSTM input_size=2048)', level='WARNING')
        
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
    
    def __init__(
            self,
            detector_weights_path: Path, word_decoder_weights_path: Path,
            conf_thres: float = 0.25, iou_thres: float = 0.45, max_det: int = 600,
            vertical_padding_ratio: float = 0.05,
            use_beam_search: bool = False,
            use_spell_checker: bool = False, vocabulary_path: Path | str | None = None
    ):
        """
        Инициализация OCR модели.
        
        Args:
            detector_weights_path: путь к весам детектора
            word_decoder_weights_path: путь к весам word decoder
            conf_thres: порог уверенности для NMS (default: 0.25)
            iou_thres: порог IoU для NMS (default: 0.45)
            max_det: максимальное количество детекций (default: 600)
            vertical_padding_ratio: процент расширения bbox по вертикали (default: 0.05 = 5%)
            use_beam_search: использовать beam search decoder вместо greedy (default: False)
            use_spell_checker: использовать spell checker для коррекции слов (default: False)
            vocabulary_path: путь к словарю для spell checker (default: None = auto)
        """
        self.detector = DetectorModel(detector_weights_path)
        self.word_decoder = CRNNModel(word_decoder_weights_path, use_beam_search=use_beam_search)

        # Параметры NMS для детектора
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        
        # Параметр адаптивного расширения bbox
        self.vertical_padding_ratio = vertical_padding_ratio
        
        # Spell checker (опционально)
        self.use_spell_checker = use_spell_checker
        if use_spell_checker:
            if vocabulary_path is None:
                # Автоматически определяем путь к словарю
                from ml.config import WORKDIR
                vocabulary_path = WORKDIR / 'ml' / 'word_decoder' / 'vocabulary.json'
            
            self.spell_checker = SpellChecker(
                vocabulary_path=vocabulary_path,
                confidence_threshold=0.7,
                max_edit_distance=2,
                min_word_length=3
            )
            stats = self.spell_checker.get_stats()
            log_event(f'\033[32mSpell checker\033[0m | vocabulary_size=\033[33m{stats["vocabulary_size"]}\033[0m', level='WARNING')
        else:
            self.spell_checker = None

    
    def _crop_word(
            self,
            img: Image.Image,
            bbox: list[float],
            orig_img_id: int | None = None,
            image_crop_idx: int | None = None
    ) -> Image.Image:
        """
        Вырезает слово из изображения по bbox с адаптивным расширением.
        
        Применяет вертикальное расширение bbox для уменьшения обрезания символов.
        
        Args:
            img: исходное изображение
            bbox: [x1, y1, x2, y2]
            orig_img_id: id исходного изображения (None для тестирования)
            image_crop_idx: inner id для слова-картинки
            
        Returns:
            cropped image с расширенным bbox
        """
        x1, y1, x2, y2 = bbox
        
        # Вычисляем высоту bbox
        bbox_height = y2 - y1
        
        # Адаптивное вертикальное расширение (5-7% от высоты)
        v_padding = int(bbox_height * self.vertical_padding_ratio)
        
        # Применяем расширение с проверкой границ изображения
        y1_expanded = max(0, int(y1 - v_padding))
        y2_expanded = min(img.height, int(y2 + v_padding))
        x1_safe = max(0, int(x1))
        x2_safe = min(img.width, int(x2))
        
        # Вырезаем с расширенным bbox
        word_img = img.crop((x1_safe, y1_expanded, x2_safe, y2_expanded))
        
        # Сохраняем crop на диск (если не тестирование)
        if orig_img_id is not None and image_crop_idx is not None:
            save_word_crop(word_img, orig_img_id, image_crop_idx)
        
        return word_img

    
    def _sort_words_by_position(self, words_data: list[dict]) -> list[dict]:
        """
        Сортирует слова по позиции: сверху вниз, слева направо.
        
        Улучшенная логика с кластеризацией строк:
            1. Вычисляем глобальную среднюю высоту bbox + tolerance (10%)
            2. Группируем слова в строки по Y координате
            3. Для каждой строки пересчитываем локальную среднюю Y
            4. Сортируем строки по средней Y, слова внутри строк по X
        
        Args:
            words_data: список словарей с ключами "text", "bbox", "confidence"
            
        Returns:
            отсортированный список
        """
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
        
        # Глобальная средняя высота + tolerance (10% вместо 50%)
        avg_height = heights.mean()
        tolerance = avg_height * 0.1  # Более строгая группировка
        
        # Сортируем по Y для группировки
        words_sorted = sorted(words_data, key=lambda w: w["center_y"])
        
        # Кластеризация строк с пересчётом локальной средней Y
        lines = []
        current_line = [words_sorted[0]]
        current_line_y = words_sorted[0]["center_y"]
        
        for word in words_sorted[1:]:
            if abs(word["center_y"] - current_line_y) < tolerance:
                # Та же строка
                current_line.append(word)
            else:
                # Новая строка - пересчитываем локальную среднюю Y
                line_y_values = [w["center_y"] for w in current_line]
                current_line_y = np.mean(line_y_values)
                
                lines.append((current_line_y, current_line))
                current_line = [word]
                current_line_y = word["center_y"]
        
        # Добавляем последнюю строку
        line_y_values = [w["center_y"] for w in current_line]
        current_line_y = np.mean(line_y_values)
        lines.append((current_line_y, current_line))
        
        # Сортируем строки по средней Y, слова внутри по X (слева направо)
        sorted_words = []
        for _, line in sorted(lines, key=lambda x: x[0]):
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
            words_data: отсортированный список слов (уже отсортированы через _sort_words_by_position)
            
        Returns:
            полный текст
        """
        if len(words_data) == 0:
            return ""
        
        # ВЕКТОРИЗАЦИЯ: Вычисляем среднюю высоту через NumPy
        heights = np.array([w["height"] for w in words_data])
        avg_height = heights.mean()
        line_tolerance = avg_height * 0.1  # Используем ту же tolerance что и в _sort_words_by_position
        
        lines = []
        current_line = [words_data[0]["text"]]
        current_line_y = words_data[0]["center_y"]
        
        for word in words_data[1:]:
            # Если на той же строке (используем текущую среднюю Y строки)
            if abs(word["center_y"] - current_line_y) < line_tolerance:
                current_line.append(word["text"])
            else:
                # Новая строка - сохраняем текущую и начинаем новую
                lines.append(" ".join(current_line))
                current_line = [word["text"]]
                current_line_y = word["center_y"]
        
        # Добавляем последнюю строку
        lines.append(" ".join(current_line))
        
        # Склеиваем строки
        full_text = "\n".join(lines)
        
        return full_text

    def _recognize_words_batch(self, word_imgs: list[Image.Image], batch_size: int = 32, return_raw: bool = False):
        """
        Распознаёт батч слов одновременно с оптимизированным паддингом (ОПТИМИЗАЦИЯ).
        
        Использует квартильную сегментацию для минимизации паддинга:
        - Сортирует изображения по ширине
        - Делит на сегменты по 25%
        - Каждый сегмент паддит до максимальной ширины внутри сегмента
        - Восстанавливает исходный порядок
        
        Args:
            word_imgs: список изображений слов
            batch_size: размер батча (default: 32)
            return_raw: возвращать сырые log_probs (для тестирования)
            
        Returns:
            список распознанных слов в исходном порядке(как на изображении)
            или (список слов, список log_probs) если return_raw=True
        """
        if len(word_imgs) == 0:
            return [] if not return_raw else ([], [])
        
        # Измеряем ширины всех изображений
        widths = [img.size[0] for img in word_imgs]
        
        # Сортируем по ширине (сохраняем индексы для восстановления порядка)
        sorted_indices = sorted(range(len(word_imgs)), key=lambda i: widths[i])
        sorted_imgs = [word_imgs[i] for i in sorted_indices]
        
        # Вычисляем размер сегмента (25% от общего количества)
        segment_size = max(1, len(sorted_imgs) // 4)  # Минимум 1 изображение в сегменте
        
        all_predictions = []
        all_log_probs = [] if return_raw else None
        
        # Обрабатываем сегментами
        for seg_start in range(0, len(sorted_imgs), segment_size):
            segment = sorted_imgs[seg_start:seg_start + segment_size]
            
            # Обрабатываем сегмент батчами
            for i in range(0, len(segment), batch_size):
                batch = segment[i:i + batch_size]
                
                # Конвертируем все изображения в grayscale
                batch_gray = [img.convert('L') for img in batch]
                
                # Трансформируем все изображения
                img_tensors = [self.word_decoder.dataset_obj.transform(img) for img in batch_gray]
                orig_widths = [t.shape[2] for t in img_tensors]
                
                # Находим максимальную ширину в БАТЧЕ (не в сегменте)
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
                batch_tensor = self.word_decoder.dataset_obj.imagenet_normalize(batch_tensor)
                
                # Перемещаем на device
                batch_tensor = batch_tensor.to(env.device)
                
                # Forward через CRNN
                self.word_decoder.model.eval()
                with torch.no_grad():
                    logits = self.word_decoder.model(batch_tensor)  # [seq_len, batch, num_classes] - RAW LOGITS
                
                # Сохраняем logits если нужно
                if return_raw:
                    all_log_probs.append(logits)
                
                # CTC декодирование (beam search или greedy)
                if self.word_decoder.use_beam_search:
                    # Beam search декодирование
                    log_probs = torch.nn.functional.log_softmax(logits, dim=2)  # [seq_len, batch, num_classes]
                    log_probs_for_beam = log_probs.transpose(0, 1).contiguous()  # [batch, seq_len, num_classes]
                    
                    # Вычисляем lengths для каждого изображения в батче
                    # lengths = ширина изображения / stride модели (примерно seq_len)
                    batch_lengths = torch.full((len(batch),), log_probs.shape[0], dtype=torch.int32, device=env.device)
                    
                    predictions = self.word_decoder.beam_search_decoder.decode(log_probs_for_beam, lengths=batch_lengths)
                else:
                    # Greedy декодирование (baseline)
                    predictions = decode_predictions(logits, self.word_decoder.dataset_obj, blank_idx=0)
                
                all_predictions.extend(predictions)
        
        # Восстанавливаем исходный порядок
        original_order_predictions = [''] * len(word_imgs)
        for i, pred in zip(sorted_indices, all_predictions):
            original_order_predictions[i] = pred
        
        # Применяем spell checker (если включён)
        if self.use_spell_checker and self.spell_checker is not None:
            corrected_predictions = []
            for pred in original_order_predictions:
                # Корректируем каждое слово
                # TODO: можно добавить confidence для каждого слова
                corrected = self.spell_checker.correct_word(pred, confidence=None)
                corrected_predictions.append(corrected)
            original_order_predictions = corrected_predictions

        if return_raw:
            return original_order_predictions, all_log_probs
        return original_order_predictions


    def _detect_words_batch(self, imgs: list[Image.Image], return_raw: bool = False):
        """
        Детекция слов на батче изображений (ОПТИМИЗАЦИЯ).
        
        Args:
            imgs: список изображений
            
        Returns:
            список кортежей (bboxes, confidences, class_ids) для каждого изображения
        """

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

        if return_raw:
            return results, preds_nms
        return results



    def forward_pass(
        self, 
        imgs: list[Image.Image],
        img_ids: list[int] | list[None],
        return_details: bool = False,
        return_raw_predictions: bool = False
    ) -> list[dict] | tuple[list[dict], dict]:
        """
        Батчинг OCR для одного или нескольких изображений.
        Чтобы подать одно изображение - обернуть в список
        
        Использует батчинг для детектора и word decoder для ускорения обработки.
        
        Args:
            imgs: список изображений
            img_ids: список id изображений в БД. Список None только для тестирования модели!
            return_details: возвращать детальную информацию
            return_raw_predictions: возвращать сырые предсказания для расчёта loss (только для тестирования)
            
        Returns:
            Если return_raw_predictions=False:
                список результатов для каждого изображения
            Если return_raw_predictions=True:
                (список результатов, dict с сырыми предсказаниями)
                raw_predictions = {
                    'crnn_log_probs': list[torch.Tensor],  # [seq_len, num_classes] для каждого слова
                    'detector_preds': list[torch.Tensor],  # [num_boxes, 6] для каждого изображения (до NMS)
                }
        """

        if len(imgs) == 0:
            if return_raw_predictions:
                return [], {'crnn_log_probs': [], 'detector_preds': []}
            return []

        # Валидация. Нельзя передать пустой список img_ids, если продакшн режим
        if isinstance(img_ids[0], type(None)) and env.app_mode == AppMode.PROD:
            raise TypeError('You cannot pass img_ids as list[None] in the production environment (app_mode=prod)')

        # Для сбора сырых предсказаний
        raw_predictions = {
            'crnn_log_probs': [],
            'detector_preds': []
        } if return_raw_predictions else None

        # 1. Детекция для всех изображений (батчинг детектора)
        if return_raw_predictions:
            batch_detections, detector_raw_preds = self._detect_words_batch(imgs, return_raw=True)
            raw_predictions['detector_preds'] = detector_raw_preds
        else:
            batch_detections = self._detect_words_batch(imgs)
        
        results = []
        
        # 2. Обрабатываем каждое изображение
        for img_id, img, (bboxes, confidences, class_ids) in zip(img_ids, imgs, batch_detections):
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
            
            # Вырезаем все слова (без сохранения на диск если img_id=None)
            word_imgs = [self._crop_word(img, bbox, img_id, idx) for idx, bbox in enumerate(bboxes)]
            
            # Распознаём батчем (батчинг word decoder)
            if return_raw_predictions:
                word_texts, word_log_probs = self._recognize_words_batch(word_imgs, return_raw=True)
                raw_predictions['crnn_log_probs'].extend(word_log_probs)
            else:
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
        
        if return_raw_predictions:
            return results, raw_predictions
        return results
