import torch
from tqdm import tqdm
from ultralytics.utils.metrics import box_iou, ap_per_class
from ultralytics.utils.nms import non_max_suppression

from ml.config import WORKDIR, env
from ml.logger_config import log_event
from ml.models import model_detector, model_detector_code

from ultralytics.utils.loss import v8DetectionLoss
from types import SimpleNamespace
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
from ml.dataclass_detector import OCRDetectorDataset
from torch.utils.data import DataLoader

from ml.utils.train_run_plots import plot_validation_metrics, plot_training_dynamics, plot_train_val_box_cls_dfl

# ======================================================================================================================
# Датасет, Даталоадеры
# ======================================================================================================================

def train_run():
    """Запуск обучения модели"""
    "Гиперпараметры"
    batch_size_train = 4  # Уменьшено для экономии памяти GPU
    batch_size_val = 4
    batch_size_test = 4
    accumulation_steps = 2  # Эффективный batch = 4 * 2 = 8
    dataload_workers = 6
    prefetch_factor = 2
    img_size = 1280
    
    print(f"Реальный batch size: {batch_size_train}")
    print(f"Эффективный batch size: {batch_size_train * accumulation_steps}")


    train_dset = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'train', 'train', img_size)
    val_dset = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'val', 'val', img_size)
    test_dset = OCRDetectorDataset(WORKDIR / 'dataset' / 'iam-form-stratified' / 'test', 'val', img_size)

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=dataload_workers,
        collate_fn=OCRDetectorDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size_val,
        num_workers=dataload_workers,
        collate_fn=OCRDetectorDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size_test,
        num_workers=dataload_workers,
        collate_fn=OCRDetectorDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )


# ======================================================================================================================
# Гиперпараметры
# ======================================================================================================================


    model_detector.to(env.device)
    loss_func = v8DetectionLoss(model_detector)
    hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5) # изменить cls на 1.0?
    loss_func.hyp = hyp

    opt = AdamW(model_detector.parameters(), lr=0.01, weight_decay=5e-4)
    lr_sched = MultiStepLR(opt, milestones=[5, 35, 50], gamma=0.1) # Обязательно сменить подход сбора last_lr при смене планировщика!


    early_stopping_mode = False
    threshold_loss = 0.01
    early_stopping = 12
    models_dir = WORKDIR / 'ml' / 'model_weights' / 'detector' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    models_dir.mkdir(parents=True)

    epochs = 60  # Полное обучение (было 5 для теста)

# ======================================================================================================================
# Обучение
# ======================================================================================================================

    log_event(f'\033[34mОбучение началось\033[0m | Эпохи: \033[33m{epochs}\033[0m')

    train_loss, val_loss, lr_list, map50_list, map5095_list, min_loss, max_map50 = [], [], [], [], [], None, None
    plateau_loss_epochs = 0
    iouv = torch.linspace(0.5, 0.95, 10).to(env.device)

    for epoch in range(1, epochs + 1):

        model_detector.train()

        last_losses_train = []
        train_loop = tqdm(train_loader, leave=False, desc=f'Training \033[34m#{epoch}\033[0m')
        
        opt.zero_grad()  # Один раз в начале эпохи
        
        for i, (img, targets, words) in enumerate(train_loop):
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
                opt.zero_grad()

            "Считаем лосс (умножаем обратно для правильного отображения)"
            total_loss = backward_tsr.detach().item() * accumulation_steps
            losses = [loss_tsr.mean().detach().item() for loss_tsr in losses_tsr]
            losses.append(total_loss)
            last_losses_train = losses
        
        # ВАЖНО: если последний батч не кратен accumulation_steps, обновляем веса
        if (i + 1) % accumulation_steps != 0:
            opt.step()
            opt.zero_grad()


        log_event(f"\033[32mTRAINING\033[0m | Epoch {epoch}, train_loss={last_losses_train[-1]:.4f}, box_loss={last_losses_train[0]:.4f}, cls_loss={last_losses_train[1]:.4f}, dfl_loss={last_losses_train[2]:.4f}")
        train_loss.append(last_losses_train)

    # ======================================================================================================================
    # Валидация
    # ======================================================================================================================

        model_detector.eval()
        with torch.no_grad():
            last_losses_val = []
            stats = []

            val_loop = tqdm(val_loader, leave=False, desc=f'Validation \033[36m#{epoch}\033[0m')
            for img, targets, words in val_loader:

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
                for i, pred in enumerate(preds_nms):
                    # Фильтруем таргеты: берем только те, где batch_idx == i
                    gt_mask = (targets[:, 0] == i)
                    gt = targets[gt_mask]  # Теперь здесь [num_gt, 6] (batch_idx, cls, x, y, w, h)

                    nl = gt.shape[0]
                    npr = pred.shape[0]
                    correct = torch.zeros(npr, len(iouv), dtype=torch.bool, device=env.device)

                    # Подготовка правильных координат GT для сопоставления

                    # В targets xywh (нормализованные), а NMS выдает xyxy (в пикселях)
                    if nl:
                        # 1. Переводим GT из xywh в xyxy
                        # 2. Денормализуем (умножаем на размер изображения, например 640)
                        h, w = img_size, img_size
                        gt_boxes = gt[:, 2:].clone()

                        # Конвертация xywh -> xyxy
                        gn = torch.tensor([w, h, w, h], device=env.device)
                        gt_boxes[:, [0, 2]] *= w # x, w
                        gt_boxes[:, [1, 3]] *= h # y, h

                        x, y, w_gt, h_gt = gt_boxes.unbind(1)
                        gt_xyxy = torch.stack([
                            x - w_gt / 2, y - h_gt / 2,
                            x + w_gt / 2, y + h_gt / 2
                        ], dim=1)
                    else:
                        gt_xyxy = torch.zeros((0, 4), device=env.device)

                    if nl:
                        # Теперь gt_xyxy имеет размерность [nl, 4]
                        iou = box_iou(pred[:, :4], gt_xyxy)
                        for j in range(len(iouv)):
                            correct[:, j] = (iou >= iouv[j]).any(1)
                    if npr > 0:
                        # stats ожидает:
                        # 1. correct [npr, 10]
                        # 2. conf [npr]
                        # 3. pred_cls [npr]
                        # 4. target_cls [nl] <-- ВАЖНО: только истинные классы изображения
                        stats.append((
                            correct.cpu(),
                            pred[:, 4].cpu(),
                            pred[:, 5].cpu(),
                            gt[:, 1].cpu()
                        ))

        lr_sched.step()
        lr = lr_sched.get_last_lr()[0]
        lr_list.append(lr)

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

        map50_list.append(map50)
        map5095_list.append(map5095)
        val_loss.append(last_losses_val)

        log_event(f"\033[34mVALIDATION\033[0m Epoch {epoch} | val_loss={last_losses_val[-1]:.4f}, box_loss={last_losses_val[0]:.4f}, cls_loss={last_losses_val[1]:.4f}, dfl_loss={last_losses_val[2]:.4f} | mAP@0.5={map50:.4f} | mAP@0.5:0.95={map5095:.4f}")

        history = {
            "general_metrics": {
                'val_loss_list': val_loss,
                'train_loss_list': train_loss,
                'map50_list': map50_list,
                'map5095_list': map5095_list,
            },
            "train_loss_last": last_losses_train,
            "val_loss_last": val_loss,
            "map50_cur": map50,
            "map5095_cur": map5095,
            "lr": lr_list
        }

        "Сохраняем модель, как только виднеется прогресс"
        min_loss = last_losses_val[-1] if min_loss is None else min_loss
        if min_loss - min_loss * threshold_loss > last_losses_val[-1]:
            min_loss = last_losses_val[-1]

            checkpoint = {
                'model_detector': model_detector_code,
                'state_model': model_detector.state_dict(),
                'state_opt': opt.state_dict(),
                'state_lr_scheduler': lr_sched.state_dict(),
                'save_epoch': epoch,
                'history': history,
            }
            torch.save(checkpoint, models_dir.joinpath(f'model_detector{epoch}.pth'))
            plateau_loss_epochs = 0
            log_event(f'На эпохе - \033[35m{epoch}\033[0m модель сохранена | Ранняя остановка обучения изменена', level='WARNING')

        "Early Stopping"
        if early_stopping_mode and plateau_loss_epochs >= early_stopping:
            log_event(f'\033[31m{'!!!' * 10} Принудительная остановка обучения, нет прогресса {'!!!' * 10}\033[0m', level='WARNING')
            raise Exception("Early Stopping")

        plateau_loss_epochs += 1

    plot_validation_metrics(history, models_dir / 'validation.png')
    plot_training_dynamics(history, models_dir / 'training.png')
    plot_train_val_box_cls_dfl(history, models_dir / 'loss_components.png')

    log_event(f'{'>>>' * 10} Обучение завершено {'<<<' * 10}')


if __name__ == '__main__':
    train_run()