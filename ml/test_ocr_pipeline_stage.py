from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou, ap_per_class

from ml.api_layer.ocr_model_driver import OCRModel
from ml.base_utils import calculate_ocr_metric
from ml.config import env, WORKDIR
from ml.logger_config import log_event
from ml.ocr_dataset_class import OCRPipelineDataset, ocr_pipeline_collate_fn
from ml.word_decoder.metrics import calculate_cer, calculate_wer


def test_run(detector_weights_path: Path | str, word_decoder_weights_path: Path, workers: int = 4, batch_size: int = 8):
    """
    Тестирование OCR пайплайна на тестовом наборе.
    
    Args:
        detector_weights_path: путь к весам детектора
        word_decoder_weights_path: путь к весам word decoder
        workers: количество воркеров для DataLoader
        batch_size: размер батча
    """
    # Выгружаем параметры модели
    detector_weights_path = Path(detector_weights_path)
    word_decoder_weights_path = Path(word_decoder_weights_path)
    
    if not detector_weights_path.exists():
        raise FileNotFoundError(f"Файл с весами Детектора не найден: {detector_weights_path}")
    if not word_decoder_weights_path.exists():
        raise FileNotFoundError(f"Файл с весами Ворд Декодера не найден: {word_decoder_weights_path}")

    conf_thres, iou_thres, max_dets, vertical_padd_ratio = 0.25, 0.45, env.max_det, 0.02
    use_sym_spell, use_beam_search = True, True
    model = OCRModel(
        detector_weights_path, word_decoder_weights_path, 
        conf_thres, iou_thres, max_dets, vertical_padd_ratio, 
        use_beam_search=use_beam_search,
        use_sym_spell=use_sym_spell,
    )

    # Загружаем тестовый датасет
    test_dset = OCRPipelineDataset(
        WORKDIR / 'dataset' / 'iam-form-stratified' / 'test',
        model.detector.dataset_obj.img_formats,
        model.detector.dataset_obj.class_to_idx,
        charset=model.word_decoder.charset
    )
    
    test_loader = DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=ocr_pipeline_collate_fn,
    )

    # Данные для метрик
    all_preds_texts = []  # Предсказанные тексты
    all_gt_texts = []  # Ground truth тексты
    all_supported_masks = []  # Маски поддерживаемых слов
    
    detector_stats = []  # Статистики детектора для mAP
    iouv = torch.linspace(0.5, 0.95, 10).to(env.device)

    log_event(f'Началось тестирование OCR пайплайна | Семплов: \033[32m{len(test_dset)}\033[0m; Spell checker: \033[33m{use_sym_spell}\033[0m; Beam Search: \033[35m{use_beam_search}\033[0m', level='WARNING')
    
    test_loop = tqdm(test_loader, leave=False, desc='Testing OCR Pipeline')
    for batch_data in test_loop:
        images = batch_data['images']  # list[PIL.Image]
        all_words = batch_data['all_words']  # list[list[str]]
        all_bboxes = batch_data['all_bboxes']  # list[list[list[float]]]
        all_supported_masks_batch = batch_data['all_supported_masks']  # list[list[bool]]
        all_img_ids = batch_data['all_img_ids']  # list[list[None]]
        
        # Forward pass через OCR модель
        # Обрабатываем каждое изображение отдельно (так как img_ids разной длины)
        for img, gt_words, gt_bboxes, supported_mask, img_ids in zip(
            images, all_words, all_bboxes, all_supported_masks_batch, all_img_ids
        ):
            # Предсказание модели
            results = model.forward_pass(
                imgs=[img],
                img_ids=img_ids,
                return_details=True
            )
            
            result = results[0]
            pred_text = result['text']
            pred_words = result['words']
            pred_bboxes = result['bboxes']
            pred_confidences = result['confidences']
            
            # Собираем предсказанный текст и ground truth
            gt_text = ' '.join(gt_words)  # Склеиваем ground truth слова
            
            all_preds_texts.append(pred_text)
            all_gt_texts.append(gt_text)
            all_supported_masks.append(supported_mask)

            # DEBUG: Выводим первые 3 семпла для проверки
            if len(all_preds_texts) <= 3:
                log_event(
                    f"\n\033[33m=== DEBUG Sample {len(all_preds_texts)} ===\033[0m\n"
                    f"GT words ({len(gt_words)}): {gt_words[:5]}...\n"
                    f"GT text: {gt_text[:100]}...\n"
                    f"Pred words ({len(pred_words)}): {pred_words[:5]}...\n"
                    f"Pred text: {pred_text[:100]}...\n"
                    f"Pred bboxes: {len(pred_bboxes)}\n"
                    f"GT bboxes: {len(gt_bboxes)}",
                    level='INFO'
                )

            # Статистики детектора (для mAP)
            # Конвертируем pred_bboxes в тензор
            if len(pred_bboxes) > 0:
                pred_bboxes_tensor = torch.tensor(pred_bboxes, device=env.device)  # [N, 4]
                pred_conf_tensor = torch.tensor(pred_confidences, device=env.device)  # [N]
                pred_cls_tensor = torch.zeros(len(pred_bboxes), device=env.device)  # [N] (все класс 0)
            else:
                pred_bboxes_tensor = torch.empty((0, 4), device=env.device)
                pred_conf_tensor = torch.empty(0, device=env.device)
                pred_cls_tensor = torch.empty(0, device=env.device)
            
            # Конвертируем gt_bboxes в тензор
            if len(gt_bboxes) > 0:
                gt_bboxes_tensor = torch.tensor(gt_bboxes, dtype=torch.float32, device=env.device)  # [M, 4]
                gt_cls_tensor = torch.zeros(len(gt_bboxes), device=env.device)  # [M] (все класс 0)
            else:
                gt_bboxes_tensor = torch.empty((0, 4), device=env.device)
                gt_cls_tensor = torch.empty(0, device=env.device)
            
            # Вычисляем IoU и correct predictions
            if len(pred_bboxes_tensor) > 0 and len(gt_bboxes_tensor) > 0:
                iou = box_iou(pred_bboxes_tensor, gt_bboxes_tensor)  # [N, M]
                
                # Для каждого порога IoU проверяем совпадение
                correct = torch.zeros(len(pred_bboxes_tensor), len(iouv), dtype=torch.bool, device=env.device)
                for j in range(len(iouv)):
                    correct[:, j] = (iou >= iouv[j]).any(1)
                
                detector_stats.append((
                    correct.cpu(),
                    pred_conf_tensor.cpu(),
                    pred_cls_tensor.cpu(),
                    gt_cls_tensor.cpu()
                ))
    
    # Подсчёт метрик
    log_event(f'\033[32m{'>>>' * 10} Закончили тестирование {'<<<' * 10}\033[0m', level='INFO')
    
    # CER, WER (сравниваем весь текст целиком)
    # ВАЖНО: Unsupported слова НЕ фильтруются - они часть реального текста
    # Модель должна их распознавать, даже если они не в charset
    # Нормализуем тексты:
    # 1. Заменяем переносы строк на пробелы
    # 2. Приводим к lowercase (модель не различает регистр)
    # 3. Удаляем лишние пробелы
    normalized_preds = []
    normalized_gts = []
    
    for pred, gt in zip(all_preds_texts, all_gt_texts):
        # Нормализация
        pred_norm = pred.replace('\n', ' ').lower().strip()
        gt_norm = gt.replace('\n', ' ').lower().strip()
        
        # Удаляем множественные пробелы
        pred_norm = ' '.join(pred_norm.split())
        gt_norm = ' '.join(gt_norm.split())
        
        normalized_preds.append(pred_norm)
        normalized_gts.append(gt_norm)
    
    if len(normalized_preds) > 0:
        cer = calculate_cer(normalized_preds, normalized_gts)
        wer = calculate_wer(normalized_preds, normalized_gts)
    else:
        cer, wer = 0.0, 0.0
        log_event('Нет данных для расчёта CER/WER', level='WARNING')
    
    # Статистика unsupported слов
    total_words = sum(len(mask) for mask in all_supported_masks)
    supported_words = sum(sum(mask) for mask in all_supported_masks)
    unsupported_words = total_words - supported_words
    
    # mAP@50, mAP@50-95
    map50, map50_95 = 0.0, 0.0
    if len(detector_stats) > 0:
        tp, conf, pred_cls, target_cls = zip(*detector_stats)
        
        # Конкатенируем
        stats_cat = [
            torch.cat(tp, 0).numpy(),
            torch.cat(conf, 0).numpy(),
            torch.cat(pred_cls, 0).numpy(),
            torch.cat(target_cls, 0).numpy()
        ]
        
        if stats_cat[0].any() or len(stats_cat[3]) > 0:
            results = ap_per_class(*stats_cat, names={0: 'word'})
            tp, fp, p, r, f1, ap, *_ = results
            
            map50 = ap[:, 0].mean() * 100  # Конвертируем в проценты
            map50_95 = ap.mean() * 100
    
    # Финальная метрика пайплайна
    pipeline_score_result = calculate_ocr_metric(
        cer=cer,
        wer=wer,
        map50=map50,
        map50_95=map50_95
    )

    
    # Логируем результаты
    log_event(
        f"\n\033[35m=== TESTING RESULTS ===\033[0m\n"
        f"Pipeline Score: \033[32m{pipeline_score_result:.2f}%\033[0m\n"
        f"Decoder: CER=\033[36m{cer:.2f}%\033[0m | WER=\033[36m{wer:.2f}%\033[0m\n"
        f"Detector: mAP@50=\033[35m{map50:.2f}%\033[0m | mAP@50-95=\033[35m{map50_95:.2f}%\033[0m\n"
        f"Total samples: \033[33m{len(test_dset)}\033[0m\n"
        f"Unsupported words: \033[31m{unsupported_words}/{total_words}\033[0m ({unsupported_words/total_words*100:.1f}%)",
        level='WARNING'
    )
    
    return {
        'pipeline_score': pipeline_score_result,
        'cer': cer,
        'wer': wer,
        'map50': map50,
        'map50_95': map50_95,
        'num_samples': len(test_dset),
        'interpretation': pipeline_score_result
    }


if __name__ == '__main__':
    # Пример запуска
    detector_weights = WORKDIR / 'ml/detector/model_weights/production_weights/model_detector47.pth'
    word_decoder_weights = WORKDIR / 'ml/word_decoder/model_weights/prod_weights_v2/model_epoch50.pt'
    
    results = test_run(
        detector_weights_path=detector_weights,
        word_decoder_weights_path=word_decoder_weights,
        workers=4,
        batch_size=8
    )
    
    print(f"\nFinal Pipeline Score: {results['pipeline_score']:.2f}%")
