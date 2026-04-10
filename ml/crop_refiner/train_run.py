from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
from ultralytics.utils.metrics import box_iou

from ultralytics.utils.ops import xywh2xyxy

from ml.config import WORKDIR, env
from ml.logger_config import log_event
from ml.crop_refiner.models import Extent2CoreRefiner, IoULoss, CIoULoss
from ml.crop_refiner.dataset_class import CropRefinerDataset



def freeze_backbone(model):
    """Замораживает backbone модели."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    log_event("Backbone заморожен")


def unfreeze_backbone(model):
    """Размораживает backbone модели."""
    for param in model.backbone.parameters():
        param.requires_grad = True
    log_event("\033[31mBackbone разморожен\033[0m")


def main():
    # ==================== Гиперпараметры ====================
    dset_dir = WORKDIR / 'dataset/HWR200/mobile_net_crops/dataset'

    models_dir = WORKDIR / 'ml' / 'crop_refiner' / 'model_weights' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    models_dir.mkdir(parents=True)
    
    batch_size = 128
    epochs = 60
    lr = 1e-3
    freeze_epochs = 10  # Количество эпох с замороженным backbone
    backbone_lr_multiplier = 0.1
    
    # Разделение датасета: 70% train, 15% val, 15% test
    train_ratio, val_ratio, test_ratio= 0.7, 0.15, 0.15
    target_size = (128, 128)
    workers = 6
    prefetch_factor = 2
    
    log_event(f"Устройство: {env.device}")
    log_event(f"Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")
    log_event(f"Freeze epochs: {freeze_epochs}")
    
    # ==================== Подготовка данных ====================
    log_event("Загрузка датасета...")
    dset = CropRefinerDataset(dset_dir, target_size=target_size, is_train=True, auto_load=True)
    
    total_samples = len(dset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size
    
    log_event(f"Всего сэмплов: {total_samples}")
    log_event(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")
    

    generator = torch.Generator().manual_seed(162894)
    train_dataset, val_dataset, test_dataset = random_split(
        dset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Создаём отдельные датасеты для val и test без аугментаций
    val_dataset_no_aug = CropRefinerDataset(
        data_dir=dset_dir,
        target_size=target_size,
        is_train=False,
    )
    val_dataset_no_aug.samples = [val_dataset_no_aug.samples[i] for i in val_dataset.indices]
    
    test_dataset_no_aug = CropRefinerDataset(
        data_dir=dset_dir,
        target_size=target_size,
        is_train=False,
    )
    test_dataset_no_aug.samples = [test_dataset_no_aug.samples[i] for i in test_dataset.indices]
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    
    val_loader = DataLoader(
        val_dataset_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    
    test_loader = DataLoader(
        test_dataset_no_aug,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
    )
    
    # ==================== Модель, Loss, Optimizer ====================
    log_event("Инициализация модели...")
    model = Extent2CoreRefiner().to(env.device)
    
    loss_func = IoULoss("XYXY")
    # loss_func = CIoULoss("XYXY")
    if freeze_epochs:
        # собираем все веса
        backbone_params = model.backbone.parameters()
        other_params = list(model.pool.parameters()) + list(model.regressor.parameters())

        opt = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr * backbone_lr_multiplier},
            {'params': other_params, 'lr': lr},
        ], weight_decay=5e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    
    # OneCycleLR scheduler
    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Замораживаем backbone на первые эпохи
    freeze_backbone(model)
    
    # ==================== Обучение ====================
    best_val_iou = 0.0
    best_val_loss = 999.0
    best_weights = ''
    thres_val = 0.02
    train_loss_list, train_iou_list, val_loss_list, val_iou_list, lr_list = [], [], [], [], []

    
    log_event("Начало обучения...")
    
    for epoch in range(1, epochs + 1):
        # Размораживаем backbone после FREEZE_EPOCHS
        if epoch == freeze_epochs + 1:
            unfreeze_backbone(model)

        model.train()
        running_loss = 0.0
        running_iou = 0.0

        train_loop = tqdm(train_loader, desc=f"Train Epoch#{epoch}", leave=False)
        for images, targets in train_loop:
            images = images.to(env.device, non_blocking=True)
            targets = targets.to(env.device, non_blocking=True)

            opt.zero_grad()

            # Forward pass
            predictions = model(images)
            predictions, targets = xywh2xyxy(predictions), xywh2xyxy(targets)
            loss = loss_func(predictions, targets)

            # Backward pass
            loss.backward()
            opt.step()
            lr_sched.step()

            # Метрики
            with torch.no_grad():
                iou = box_iou(predictions, targets).mean()

            running_loss += loss.item()
            running_iou += iou.item()

            current_lr = opt.param_groups[0]['lr']
            lr_list.append(current_lr)

            train_loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou.item():.4f}'
            })

        train_loss = running_loss / len(train_loader)
        train_iou = running_iou / len(train_loader)
        train_loss_list.append(train_loss)
        train_iou_list.append(train_iou)

        log_event(f"\033[32mTraining\033[0m \033[34m{epoch}\033[0m | train_loss: \033[33m{train_loss:.4f}\033[0m; train_iou: \033[35m{train_iou:.4f}\033[0m; lr: \033[31m{lr_list[-1]:.6f}\033[0m")

        "Validation"
        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_iou = 0.0

            val_loop = tqdm(val_loader, desc=f"Validation Epoch#{epoch}", leave=False)
            for images, targets in val_loop:
                images = images.to(env.device, non_blocking=True)
                targets = targets.to(env.device, non_blocking=True)

                # Forward pass
                predictions = model(images)
                predictions, targets = xywh2xyxy(predictions), xywh2xyxy(targets)
                loss = loss_func(predictions, targets)

                # Метрики
                iou = box_iou(predictions, targets).mean()

                running_loss += loss.item()
                running_iou += iou.item()

                val_loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou.item():.4f}'
                })

            val_loss = running_loss / len(val_loader)
            val_iou = running_iou / len(val_loader)
            val_loss_list.append(val_loss)
            val_iou_list.append(val_iou)


        # Логирование
        log_event(f"\033[35mValidation\033[0m \033[34m{epoch}\033[36m | val_loss: \033[35m{val_loss:.4f}\033[0m; val_iou: \033[34m{val_iou:.4f}\033[0m")
        

        
        # Сохранение лучшей модели
        if val_loss < best_val_loss + best_val_loss * thres_val:
            best_val_iou = val_iou
            best_val_loss = val_loss
            best_weights = f'crop_refiner{epoch}.pt'

            history = {
                'train_loss_cur': train_loss,
                'train_loss_list': train_loss_list,
                'train_iou_cur': train_iou,
                'train_iou_list': train_iou_list,
                'val_loss_cur': val_loss,
                'val_loss_list': val_loss_list,
                'val_iou_cur': val_iou,
                'val_iou_list': val_iou_list,
                'lr_list': lr_list,
            }
            checkpoint = {
                'epoch_cur': epoch,
                'epochs': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': lr_sched.state_dict(),
                'history': history,
            }
            save_path = models_dir.joinpath(f'crop_refiner{epoch}.pt')
            try:
                tmp_file = models_dir.joinpath(f'crop_refiner{epoch}.tmp')
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

    log_event(f"{">>>" * 10} Обучение завершено! {"<<<" * 10}")
    
    # ==================== Тестирование ====================
    log_event("Загрузка лучших весов для тестирования...")
    if best_weights:
        checkpoint = torch.load(models_dir / best_weights, weights_only=False, map_location=env.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        log_event('Нет весов для тестирования', level='WARNING')
        return
    

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_iou = 0.0

        test_loop = tqdm(test_loader, desc=f"Testing Epoch", leave=False)
        for images, targets in test_loop:
            images = images.to(env.device, non_blocking=True)
            targets = targets.to(env.device, non_blocking=True)

            # Forward pass
            predictions = model(images)
            predictions, targets = xywh2xyxy(predictions), xywh2xyxy(targets)
            loss = loss_func(predictions, targets)

            # Метрики
            iou = box_iou(predictions, targets).mean()

            running_loss += loss.item()
            running_iou += iou.item()

            test_loop.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou.item():.4f}'
            })

        test_loss = running_loss / len(val_loader)
        test_iou = running_iou / len(val_loader)
    
    log_event(f"\033[31mTesting\033[0m | test_loss: {test_loss:.4f}, test_iou: \033[32m{test_iou:.4f}\033[0m")


if __name__ == '__main__':
    main()
