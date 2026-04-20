import os
from pathlib import Path

import torch
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils.nms import non_max_suppression

from ml.config import WORKDIR, env
from ml.detector.dataset_class.hwr200_dataset import HWR200DetectorDataset
from ml.logger_config import log_event
from ml.detector.models import WordDetector, model_detector_code

from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR
from datetime import datetime
from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset
from torch.utils.data import DataLoader, Dataset

from ml.detector.utils.train_run_plots import plot_loss_dynamics, plot_metrics_dynamics, plot_lr_chronology


# ======================================================================================================================
# Датасет, Даталоадеры
# ======================================================================================================================

def train_run(
        train_dset: Dataset, val_dset: Dataset, models_dir: Path | str,

        pretrained_weights: str = None,

        epochs: int = 60,
):
    """"""
    "Гиперпараметры"
    batch_size_train = 4  # Уменьшено для экономии памяти GPU
    batch_size_val = 4
    accumulation_steps = 2  # Эффективный batch = 4 * 2 = 8
    dataload_workers = 4
    prefetch_factor = 2
    img_size = 1280
    
    log_event(f"Реальный batch size: {batch_size_train}")
    log_event(f"Эффективный batch size: {batch_size_train * accumulation_steps}")



    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=dataload_workers,
        collate_fn=train_dset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size_val,
        num_workers=dataload_workers,
        collate_fn=val_dset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    log_event(f"Семплов в \033[33mТрейн Датасете: \033[31m{len(train_dset)}\033[0m")
    log_event(f"Семплов в \033[32mВал Датасете: \033[36m{len(val_dset)}\033[0m")

# ======================================================================================================================
# Гиперпараметры
# ======================================================================================================================

    model_detector = WordDetector()


    "Transfer Learning"
    freeze_backbone, freeze_neck, freeze_head = False, False, False

    if pretrained_weights:
        if os.path.exists(pretrained_weights):
            log_event(f'Загрузка предобученных весов из: \033[36m{pretrained_weights}\033[0m')
            checkpoint = torch.load(pretrained_weights, map_location=env.device, weights_only=False)
            model_detector.load_state_dict(checkpoint['state_model'])
            log_event('Веса успешно загружены для \033[32mtransfer learning\033[0m')
        else:
            log_event(f'Файл весов не найден: \033[31m{pretrained_weights}\033[0m', level='ERROR')
            raise FileNotFoundError(f'Pretrained weights not found: {pretrained_weights}')
    
    # Заморозка backbone для transfer learning
    if pretrained_weights:
        log_event('Заморозка YOLOv8n', level='INFO')

        for i, layer in enumerate(model_detector.model):
            is_backbone = i <= 9
            is_neck = 9 < i < len(model_detector.model) - 1
            is_head = i == len(model_detector.model) - 1

            if (freeze_backbone and is_backbone) or (freeze_neck and is_neck) or (freeze_head and is_head):
                for param in layer.parameters():
                    param.requires_grad = False


        # Подсчёт замороженных/обучаемых параметров
        frozen_params = sum(p.numel() for p in model_detector.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model_detector.parameters() if p.requires_grad)
        log_event(f'Замороженные параметры: \033[36m{frozen_params:,}\033[0m', level='INFO')
        log_event(f'Обучаемые параметры: \033[32m{trainable_params:,}\033[0m', level='INFO')
        log_event(f'Статусы обучения | \033[34mBackbone\033[0m: {not freeze_backbone}; \033[35mNeck\033[0m: {not freeze_neck}; \033[31mHead\033[0m: {not freeze_head}', level='WARNING')

    model_detector.to(env.device)
    loss_func = v8DetectionLoss(model_detector)
    hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    loss_func.hyp = hyp

    if pretrained_weights:
        param_list = []
        if not freeze_backbone:
            backbone_params = []
            for layer in model_detector.model[:10]:
                backbone_params.extend(layer.parameters())
            param_list.append(
                {"params": backbone_params, 'lr': 0.00075}
            )
        if not freeze_neck:
            neck_params = []
            for layer in model_detector.model[10:len(model_detector.model) - 1]:
                neck_params.extend(layer.parameters())
            param_list.append(
                {"params": neck_params, 'lr': 0.00180}
            )
        if not freeze_head:
            param_list.append(
                {"params": list(model_detector.model[-1].parameters()), 'lr': 0.002}
            )
        opt = AdamW(param_list, weight_decay=5e-4)
    else:
        opt = AdamW(model_detector.parameters(), lr=0.001, weight_decay=5e-4)

        # lr_sched = MultiStepLR(opt, milestones=[5, 35, 50], gamma=0.1) # Обязательно сменить подход сбора last_lr при смене планировщика!
    lr_sched = OneCycleLR(
        opt,
        max_lr=0.001,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,
        anneal_strategy='cos',
    )


    early_stopping_mode = True
    threshold_loss, threshold_metric = 0.01, 0.05 # агрессивный порог в начале
    max_metric = 0.00
    early_stopping = 20
    models_dir.mkdir(parents=True, exist_ok=True)



# ======================================================================================================================
# Обучение
# ======================================================================================================================

    log_event(f'\033[34mОбучение началось\033[0m | Эпохи: \033[33m{epochs}\033[0m')

    train_loss, val_loss, lr_list, map50_list, map5095_list, min_loss, max_map50 = [], [], [], [], [], None, None
    plateau_loss_epochs = 0
    iouv = torch.linspace(0.5, 0.95, 10).to(env.device)

    for epoch in range(1, epochs + 1):

        "Управление порогом улучшения метрики"
        if epoch > (epochs * 0.8):  # Последние 20% эпох сохраняем при малейшем улучшении
            threshold_metric = 0.02

        model_detector.train()

        last_losses_train = []
        train_loop = tqdm(train_loader, leave=False, desc=f'Training \033[34m#{epoch}\033[0m')
        
        opt.zero_grad()  # Один раз в начале эпохи
        
        for i, (img, targets) in enumerate(train_loop):
            "Предсказание модели на батч"
            img = img.to(env.device, non_blocking=True)
            targets = targets.to(env.device, non_blocking=True)
            batch_dict = {
                "batch_idx": targets[:, 0],
                "cls": targets[:, 1].long(),
                "bboxes": targets[:, 2:]
            }

            preds = model_detector(img)
            total_loss_tsr, losses_tsr = loss_func(preds, batch_dict)
            
            # ВАЖНО: делим loss на accumulation_steps для правильного масштаба градиентов
            backward_tsr = total_loss_tsr.mean() / accumulation_steps

            backward_tsr.backward()  # Накапливаем градиенты
            
            # Обновляем веса только каждые accumulation_steps батчей
            if (i + 1) % accumulation_steps == 0:
                opt.step()
                lr_sched.step()
                opt.zero_grad()

            "Считаем лосс (умножаем обратно для правильного отображения)"
            total_loss = backward_tsr.detach().item() * accumulation_steps
            losses = [loss_tsr.mean().detach().item() for loss_tsr in losses_tsr]
            losses.append(total_loss)
            last_losses_train = losses
        
        # ВАЖНО: если последний батч не кратен accumulation_steps, обновляем веса
        if (i + 1) % accumulation_steps != 0:
            opt.step()
            lr_sched.step()
            opt.zero_grad()


        log_event(f"\033[32mTRAINING\033[0m | Epoch {epoch} | train_loss=\033[33m{last_losses_train[-1]:.4f}\033[0m, box_loss={last_losses_train[0]:.4f}, cls_loss={last_losses_train[1]:.4f}, dfl_loss={last_losses_train[2]:.4f}")
        train_loss.append(last_losses_train)

    # ======================================================================================================================
    # Валидация
    # ======================================================================================================================

        model_detector.eval()
        with torch.no_grad():
            last_losses_val = []
            stats = []

            val_loop = tqdm(val_loader, leave=False, desc=f'Validation \033[36m#{epoch}\033[0m')
            for img, targets in val_loop:

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

                last_losses_val = losses

                preds_nms = non_max_suppression(
                    preds,
                    conf_thres=0.25,
                    iou_thres=0.45,
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
                        cy + bh / 2   # y2
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
                        gt[:, 1].cpu()     # target class
                    ))

        "MultiStep LR Scheduler"
        # lr_sched.step()
        # lr = lr_sched.get_last_lr()[0]
        # lr_list.append(lr)

        "OneCycleLR Scheduler"
        lr = opt.param_groups[0]['lr']
        lr_list.append(lr)

        "Подсчёт метрик за эпоху"
        map50, map5095 = 0.0, 0.0
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

        map50_list.append(map50)
        map5095_list.append(map5095)
        val_loss.append(last_losses_val)

        log_event(f"\033[34mVALIDATION\033[0m Epoch {epoch} | val_loss=\033[31m{last_losses_val[-1]:.4f}\033[0m | mAP@0.5=\033[33m{map50:.4f}\033[0m | mAP@0.5:0.95=\033[36m{map5095:.4f}\033[0m | LR=\033[35m{lr}\033[0m | box_loss={last_losses_val[0]:.4f}, cls_loss={last_losses_val[1]:.4f}, dfl_loss={last_losses_val[2]:.4f}")

        history = {
            "general_metrics": {
                'val_loss_list': val_loss,
                'train_loss_list': train_loss,
                'map50_list': map50_list,
                'map5095_list': map5095_list,
            },
            "train_loss_last": last_losses_train,
            "val_loss_last": last_losses_val,
            "map50_cur": map50,
            "map5095_cur": map5095,
            "lr": lr_list
        }

        "Сохраняем модель, как только виднеется прогресс"
        # min_loss = last_losses_val[-1] if min_loss is None else min_loss
        cur_metric = (map5095 + map50) / 2
        if cur_metric * threshold_metric + cur_metric > max_metric:
        # if min_loss - min_loss * threshold_loss > last_losses_val[-1]:
        #     min_loss = last_losses_val[-1]
            max_metric = cur_metric

            checkpoint = {
                'model_code': model_detector_code,
                'state_model': model_detector.state_dict(),
                'state_opt': opt.state_dict(),
                'state_lr_scheduler': lr_sched.state_dict(),
                'save_epoch': epoch,
                'img_size': img_size,
                'history': history,
            }
            save_path = str(models_dir.joinpath(f'model_detector{epoch}.pt'))
            try:
                tmp_file = models_dir.joinpath(f'model_detector{epoch}.tmp')
                torch.save(checkpoint, save_path)
                if os.path.exists(tmp_file) and os.path.getsize(tmp_file) > 0:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    os.rename(tmp_file, save_path)
                    log_event(f"Успешно сохранено: {save_path} ({os.path.getsize(save_path) / 1e6:.2f} MB)")
                else:
                    log_event(f"Файл {tmp_file} пуст!", level='ERROR')
            except Exception as e:
                log_event(f"Критическая ошибка при сохранении: {e}", level='CRITICAL')

            plateau_loss_epochs = 0
            log_event(f'На эпохе - \033[35m{epoch}\033[0m модель сохранена | Ранняя остановка обучения изменена', level='WARNING')

        "Early Stopping"
        if early_stopping_mode and plateau_loss_epochs >= early_stopping:
            log_event(f'\033[31m{'!!!' * 10} Принудительная остановка обучения, нет прогресса {'!!!' * 10}\033[0m', level='WARNING')
            plot_loss_dynamics(history, models_dir / 'loss_distribution.png')
            plot_metrics_dynamics(history, models_dir / 'metrics.png')
            plot_lr_chronology(history, models_dir / 'lr_chronology.png')
            raise Exception("Early Stopping")

        plateau_loss_epochs += 1

    plot_loss_dynamics(history, models_dir / 'loss_distribution.png')
    plot_metrics_dynamics(history, models_dir / 'metrics.png')
    plot_lr_chronology(history, models_dir / 'chronology.png')

    log_event(f'\033[34m{'>>>' * 10} Обучение завершено {'<<<' * 10}\033[0m')

if __name__ == '__main__':
    epochs = 60
    img_size = 1280
    # models_dir = WORKDIR / 'ml' / 'detector' / 'model_weights' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    models_dir = WORKDIR / 'ml' / 'detector' / 'model_weights' / 'transfer_learning_core_bounding_v1'


    "IAM Handwrite Dataset"
    # train_dset_obj = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'train', 'train', img_size)
    # val_dset_obj = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'val', 'val', img_size)

    "HWR200 Dataset"
    train_dset_obj = HWR200DetectorDataset(WORKDIR / 'dataset' / 'HWR200' / 'simplified' / 'train_248', 'train', img_size=img_size)
    val_dset_obj = HWR200DetectorDataset(WORKDIR / 'dataset' / 'HWR200' / 'simplified' / 'val_60', 'val', img_size=img_size)

    transfer_learning_weights = WORKDIR / 'ml/detector/model_weights/production_weights_v2/model_detector52.pt'
    train_run(
        train_dset_obj, val_dset_obj, models_dir,
        # pretrained_weights=transfer_learning_weights,
        epochs=epochs,
    )
