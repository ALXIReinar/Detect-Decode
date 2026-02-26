from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.nms import non_max_suppression

from ml.config import WORKDIR, env
from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset
from ml.logger_config import log_event
from ml.detector.models import model_detector
from ml.detector.utils.args_parser_test_stage import parse_args


# ======================================================================================================================
# Тестирование
# ======================================================================================================================

def test_run(weights_path: Path | str, batch_size: int = 4, img_size: int = 1280, workers: int = 2, conf_thres: float = 0.25, iou_thres: float = 0.45):
    """
    Тестирование модели детектора.
    
    Args:
        weights_path: Путь к весам модели (.pth файл)
        batch_size: Размер батча для тестирования
        img_size: Размер входного изображения
        workers: Количество workers для DataLoader
        conf_thres: Порог confidence для NMS
        iou_thres: Порог IoU для NMS
    """
    batch_size_test = batch_size
    dataload_workers = workers
    prefetch_factor = 2

    test_dset = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'test', 'val', img_size)
    test_loader = DataLoader(
        dataset=test_dset,
        batch_size=batch_size_test,
        num_workers=dataload_workers,
        collate_fn=OCRDetectorDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )

    "Выгружаем веса модели"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
    
    log_event(f'Загрузка весов из: \033[36m{weights_path}\033[0m', level='WARNING')
    
    model_detector.to(env.device)
    model_weights = torch.load(weights_path, weights_only=False, map_location=env.device)['state_model']
    model_detector.load_state_dict(model_weights)
    
    log_event(f'✅ Веса загружены успешно', level='WARNING')

    "Лосс функция"
    loss_func = v8DetectionLoss(model_detector)
    hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    loss_func.hyp = hyp
    iouv = torch.linspace(0.5, 0.95, 10).to(env.device)

    log_event(f'Началось тестирование модели с весами \033[36m{weights_path}\033[0m', level='WARNING')
    model_detector.eval()
    with torch.no_grad():
        last_losses_test = []
        stats = []

        test_loop = tqdm(test_loader, leave=False, desc=f'Validation \033[36m#-1\033[0m')
        for img, targets in test_loop:

            img = img.to(env.device, non_blocking=True)
            targets = targets.to(env.device, non_blocking=True)

            batch_dict = {
                "batch_idx": targets[:, 0],
                "cls": targets[:, 1].long(),
                "bboxes": targets[:, 2:]
            }

            preds = model_detector(img)
            total_loss_tsr, losses_tsr = loss_func(preds, batch_dict)

            total_loss = total_loss_tsr.mean().detach().item()
            losses = [loss_tsr.mean().detach().item() for loss_tsr in losses_tsr]
            losses.append(total_loss)

            last_losses_test = losses

            preds_nms = non_max_suppression(
                preds,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=True,
                max_det=600,
                nc=model_detector.nc
            )

            "Метрики"
            # Подготавливаем GT bbox в формате xyxy (пиксели)
            h, w = img_size, img_size
            for i, pred in enumerate(preds_nms):
                if pred.shape[0] == 0:  # Нет предсказаний
                    continue

                # Фильтруем GT для текущего изображения
                gt_mask = (targets[:, 0] == i)
                gt = targets[gt_mask]  # [num_gt, 6]: (batch_idx, cls, cx, cy, w, h)

                nl = gt.shape[0]  # Количество GT bbox
                npr = pred.shape[0]  # Количество предсказанных bbox

                if nl == 0:  # Нет GT bbox
                    continue

                # Конвертируем GT из нормализованного xywh в xyxy (пиксели)
                gt_boxes = gt[:, 2:].clone()  # [nl, 4]: (cx, cy, w, h)

                # Денормализация и конвертация xywh -> xyxy (векторизованно)
                gt_boxes[:, [0, 2]] *= w  # cx, w
                gt_boxes[:, [1, 3]] *= h  # cy, h

                cx, cy, bw, bh = gt_boxes.unbind(1)
                gt_xyxy = torch.stack([
                    cx - bw / 2,  # x1
                    cy - bh / 2,  # y1
                    cx + bw / 2,  # x2
                    cy + bh / 2  # y2
                ], dim=1)  # [nl, 4]

                # Вычисляем IoU между всеми предсказаниями и GT
                iou = box_iou(pred[:, :4], gt_xyxy)  # [npr, nl]

                # Для каждого порога IoU проверяем, есть ли совпадение
                correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=env.device)
                for j in range(len(iouv)):
                    correct[:, j] = (iou >= iouv[j]).any(1)  # [npr]

                # Сохраняем статистики
                stats.append((
                    correct.cpu(),
                    pred[:, 4].cpu(),  # confidence
                    pred[:, 5].cpu(),  # predicted class
                    gt[:, 1].cpu()  # target class
                ))

    "Подсчёт метрик за эпоху"
    if len(stats):
        tp, conf, pred_cls, target_cls = zip(*stats)

        # Конкатенируем (tp — это список тензоров [npr, 10], остальные [npr] или [nl])
        stats_cat = [
            torch.cat(tp, 0).numpy(),
            torch.cat(conf, 0).numpy(),
            torch.cat(pred_cls, 0).numpy(),
            torch.cat(target_cls, 0).numpy()
        ]

        if stats_cat[0].any() or len(stats_cat[3]) > 0:
            results = ap_per_class(*stats_cat, names=model_detector.names)
            tp, fp, p, r, f1, ap, *_ = results

            map50 = ap[:, 0].mean()
            map5095 = ap.mean()
        else:
            map50, map5095 = 0.0, 0.0

    else:
        map50, map5095 = 0.0, 0.0

    log_event(
        f"\n\033[33m{'='*80}\033[0m\n"
        f"\033[33mТЕСТИРОВАНИЕ ЗАВЕРШЕНО\033[0m\n"
        f"\033[33m{'='*80}\033[0m\n"
        f"Веса: {weights_path.name}\n"
        f"Test Loss: \033[31m{last_losses_test[-1]:.4f}\033[0m\n"
        f"  - Box Loss: {last_losses_test[0]:.4f}\n"
        f"  - Cls Loss: {last_losses_test[1]:.4f}\n"
        f"  - DFL Loss: {last_losses_test[2]:.4f}\n"
        f"mAP@0.5: \033[33m{map50:.4f}\033[0m ({map50*100:.2f}%)\n"
        f"mAP@0.5:0.95: \033[36m{map5095:.4f}\033[0m ({map5095*100:.2f}%)\n"
        f"\033[33m{'='*80}\033[0m",
        level='WARNING'
    )
    
    return {
        'test_loss': last_losses_test[-1],
        'box_loss': last_losses_test[0],
        'cls_loss': last_losses_test[1],
        'dfl_loss': last_losses_test[2],
        'map50': map50,
        'map5095': map5095
    }




if __name__ == '__main__':
    args = parse_args()
    
    weights_path_arg = Path(args.weights)
    
    # Если путь относительный, делаем его относительно WORKDIR
    if not weights_path_arg.is_absolute():
        weights_path_arg = WORKDIR / weights_path_arg

    # Запуск
    res = test_run(
        weights_path=weights_path_arg,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres
    )
