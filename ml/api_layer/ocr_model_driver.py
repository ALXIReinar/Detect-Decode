from importlib import resources
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from symspellpy import Verbosity, SymSpell
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


class CRNNModel:
    def __init__(self, weights_path: Path, use_beam_search: bool = False):
        """
        Args:
            weights_path: путь к весам CRNN модели
            use_beam_search: использовать beam search decoder вместо greedy (default: False)
        """
        self.padding_value = None
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

        if self.padding_value is None:
            raise ValueError('CRNN weights must include padding_value!!!')

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
        model_inner_params = model_hyperparams['model_params']

        self.img_height = model_hyperparams['img_height']
        self.charset = model_hyperparams['charset']
        self.beam_size = model_hyperparams.get('beam_size', 10)
        self.padding_value = model_inner_params.get('padding_value', 1.0)

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
            use_sym_spell: bool = True, word_confidence_threshold: float = 0.7, max_edit_distance: int = 2,
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
            use_sym_spell: использовать symspellpy для коррекции слов (default: True)
        """
        "Модели"
        self.detector = DetectorModel(detector_weights_path)
        self.word_decoder = CRNNModel(word_decoder_weights_path, use_beam_search=use_beam_search)

        # Параметры NMS для детектора
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

        # Adaptive Padding Bbox
        self.vertical_padding_ratio = vertical_padding_ratio

        "Параметры для SymSpell"
        self.use_sym_spell = use_sym_spell
        if use_sym_spell:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
            self.word_conf_thres = word_confidence_threshold
            self.max_edit_distance = max_edit_distance

            "Загружаем встроенный словарь (82k английских слов с частотностью)"
            dictionary_path = resources.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
            if not self.sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1):
                raise FileNotFoundError('Файл частотности слов (frequency_dictionary_en_82_765.txt) для symspellpy checker не найден')
            "Файл биграмм"
            # bigram_path = importlib.resources.files("symspellpy") / "frequency_bigramdictionary_en_243_342.txt"

            log_event(f"Используется \033[36mSymSpellPy Checker\033[0m | vocabulary_words: \033[33m{len(self.sym_spell.words)}\033[0m", level='WARNING')
        else:
            self.sym_spell = None



    def correct_words_data(self, words_data: list[str], confidences: list[float]):
        """
        Корректирует слова с использованием SymSpell с учётом confidence.
        
        Args:
            words_data: список слов для коррекции
            confidences: список confidence для каждого слова
            
        Returns:
            список скорректированных слов
        """
        corrections_count = 0
        skipped_high_conf = 0
        skipped_short = 0
        no_suggestions = 0
        same_word = 0
        low_conf_not_corrected = []  # Слова с низкой confidence которые не были скорректированы
        
        for i, (word, conf) in enumerate(zip(words_data, confidences)):
            # Если word_decoder очень уверен, отдаём без корректировок
            if conf > self.word_conf_thres:
                skipped_high_conf += 1
                continue

            # Если слово - просто цифры или очень короткое, тоже можно пропустить
            if not word.isalpha() or len(word) < 2:
                skipped_short += 1
                continue

            # Ищем предложения по исправлению слов
            suggestions = self.sym_spell.lookup(
                word.lower(),
                Verbosity.CLOSEST,
                max_edit_distance=self.max_edit_distance
            )

            if not suggestions:
                no_suggestions += 1
                if conf < 0.7:  # Низкая confidence но нет suggestions
                    low_conf_not_corrected.append((word, conf, 'no_suggestions'))
                continue
            
            # Берем лучший вариант
            best_guess = suggestions[0].term
            
            # Проверяем что это действительно коррекция (не то же самое слово)
            if best_guess == word.lower():
                same_word += 1
                if conf < 0.7:  # Низкая confidence но слово уже правильное
                    low_conf_not_corrected.append((word, conf, 'same_word'))
                continue
            
            # Сохраняем регистр оригинального слова
            if word[0].isupper():
                corrected_word = best_guess.capitalize()
            else:
                corrected_word = best_guess
            
            # DEBUG: Логируем первые 5 коррекций
            if corrections_count < 5:
                log_event(
                    f"SymSpell correction: '{word}' (conf={conf:.3f}) -> '{corrected_word}'",
                    level='INFO'
                )
            
            words_data[i] = corrected_word
            corrections_count += 1

        # Логируем статистику
        log_event(
            f"SymSpell: corrected={corrections_count}, "
            f"skipped_high_conf={skipped_high_conf}, "
            f"skipped_short={skipped_short}, "
            f"no_suggestions={no_suggestions}, "
            f"same_word={same_word}, "
            f"total_words={len(words_data)}",
            level='INFO'
        )
        
        # Логируем первые 5 слов с низкой confidence которые не были скорректированы
        if low_conf_not_corrected:
            log_event(
                f"Low conf not corrected (first 5): {low_conf_not_corrected[:5]}",
                level='INFO'
            )

        return words_data


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
        
        # Адаптивное вертикальное расширение (2-5% от высоты)
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

        Улучшенная логика с вертикальным перекрытием bbox:
            1. Сортируем слова по верхней границе (y_top)
            2. Группируем в строки по вертикальному перекрытию (IoU по Y)
            3. Если перекрытие > 50% высоты слова → та же строка
            4. Сортируем строки по Y, слова внутри строк по X

        Args:
            words_data: список словарей с ключами "text", "bbox", "confidence"

        Returns:
            отсортированный список
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
        
        # Вычисляем среднюю высоту через список (уже есть в словарях)
        heights = [w["height"] for w in words_data]
        avg_height = sum(heights) / len(heights)
        line_tolerance = avg_height * 0.5  # 50% для определения новой строки
        
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

        # Отдаём весь текст
        return  "\n".join(lines)

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
            (список распознанных слов, список confidence) в исходном порядке
            или (список слов, список confidence, список log_probs) если return_raw=True
        """
        if len(word_imgs) == 0:
            return ([], []) if not return_raw else ([], [], [])
        
        # Измеряем ширины всех изображений
        widths = [img.size[0] for img in word_imgs]
        
        # Сортируем по ширине (сохраняем индексы для восстановления порядка)
        sorted_indices = sorted(range(len(word_imgs)), key=lambda i: widths[i])
        sorted_imgs = [word_imgs[i] for i in sorted_indices]
        
        # Вычисляем размер сегмента (25% от общего количества)
        segment_size = max(1, len(sorted_imgs) // 4)  # Минимум 1 изображение в сегменте
        
        all_predictions = []
        all_confidences = []
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
                        # Паддим справа до max_width
                        padding = (0, max_width - w, 0, 0)
                        padded_img = F.pad(img_tensor, padding, mode='constant', value=self.word_decoder.padding_value)
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
                
                # Вычисляем log_probs для confidence
                log_probs = torch.nn.functional.log_softmax(logits, dim=2)  # [seq_len, batch, num_classes]
                
                # Сохраняем logits если нужно
                if return_raw:
                    all_log_probs.append(logits)
                
                # CTC декодирование
                if self.word_decoder.use_beam_search:
                    # Beam search декодирование
                    log_probs_for_beam = log_probs.transpose(0, 1).contiguous()  # [batch, seq_len, num_classes]
                    
                    # Вычисляем lengths для каждого изображения в батче
                    batch_lengths = torch.full((len(batch),), log_probs.shape[0], dtype=torch.int32, device=env.device)
                    
                    predictions = self.word_decoder.beam_search_decoder.decode(log_probs_for_beam, lengths=batch_lengths)
                else:
                    # Greedy декодирование (baseline)
                    predictions = decode_predictions(logits, self.word_decoder.dataset_obj, blank_idx=0)
                
                # Вычисляем confidence для каждого слова
                # Confidence = средняя вероятность предсказанных символов (без blank)
                probs = torch.exp(log_probs)  # [seq_len, batch, num_classes]
                
                for batch_idx, pred_text in enumerate(predictions):
                    if len(pred_text) == 0:
                        # Пустое предсказание → низкая уверенность
                        all_confidences.append(0.0)
                    else:
                        # Получаем индексы предсказанных символов
                        pred_indices = [self.word_decoder.dataset_obj.char_to_idx.get(char, 0) for char in pred_text]
                        
                        # Вычисляем среднюю вероятность предсказанных символов
                        # Берём максимальную вероятность на каждом timestep и усредняем
                        max_probs = probs[:, batch_idx, :].max(dim=1)[0]  # [seq_len]
                        confidence = max_probs.mean().item()
                        
                        all_confidences.append(confidence)
                
                all_predictions.extend(predictions)
        
        # Восстанавливаем исходный порядок
        original_order_predictions = [''] * len(word_imgs)
        original_order_confidences = [0.0] * len(word_imgs)
        
        for i, (pred, conf) in zip(sorted_indices, zip(all_predictions, all_confidences)):
            original_order_predictions[i] = pred
            original_order_confidences[i] = conf
        
        # Применяем symspellpy (если включён) с учётом confidence
        if self.use_sym_spell:
            original_order_predictions = self.correct_words_data(original_order_predictions, original_order_confidences)

        if return_raw:
            return original_order_predictions, original_order_confidences, all_log_probs
        return original_order_predictions, original_order_confidences


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
                word_texts, word_confidences, word_log_probs = self._recognize_words_batch(word_imgs, return_raw=True)
                raw_predictions['crnn_log_probs'].extend(word_log_probs)
            else:
                word_texts, word_confidences = self._recognize_words_batch(word_imgs)
            
            # Формируем words_data
            words_data = [
                {
                    "text": text,
                    "bbox": bbox,
                    "confidence": conf,
                    "class_id": cls_id,
                    "word_confidence": word_conf  # Confidence от CRNN
                }
                for text, bbox, conf, cls_id, word_conf in zip(word_texts, bboxes, confidences, class_ids, word_confidences)
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
