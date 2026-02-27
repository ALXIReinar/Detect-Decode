import torch
from torch.nn import CTCLoss
from tqdm import tqdm

from ml.config import WORKDIR, env
from ml.logger_config import log_event


from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from datetime import datetime
from torch.utils.data import DataLoader

from ml.word_decoder.dataset_class.dataclass_word_decoder import CRNNWordDataset
from ml.word_decoder.models import CRNNWordEncoder, model_word_encoder_code
from ml.word_decoder.metrics import calculate_cer, calculate_wer, decode_predictions
from ml.word_decoder.utils import plot_lr_chronology, plot_loss_dynamics, plot_metrics_dynamics


# ======================================================================================================================
# Датасет, Даталоадеры
# ======================================================================================================================

def train_run():
    """Запуск обучения модели"""
    "Гиперпараметры"
    "Параметры градиентного аккумулирования"
    gradient_accumulation_mode = False
    # batch_size_train = 4  # Уменьшено для экономии памяти GPU
    accumulation_steps = 2  # Эффективный batch = 4 * 2 = 8

    batch_size_train = 64
    batch_size_val = 64
    dataload_workers = 6
    prefetch_factor = 2
    img_height = 64
    

    base_dataset_path = WORKDIR / 'dataset' / 'iam-words'
    train_dset = CRNNWordDataset(base_dataset_path / 'train', base_dataset_path / 'charset.txt', img_height, 'train')
    val_dset = CRNNWordDataset(base_dataset_path / 'val', base_dataset_path / 'charset.txt', img_height, 'val')

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=dataload_workers,
        collate_fn=CRNNWordDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=batch_size_val,
        num_workers=dataload_workers,
        collate_fn=CRNNWordDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        drop_last=True
    )


# ======================================================================================================================
# Гиперпараметры
# ======================================================================================================================

    epochs = 60

    hidden_size = 384
    lstm_layers = 3
    rnn_dropout = 0.5
    lstm_dropout = 0.5
    pretrained_backbone = False
    num_classes = len(train_dset.charset)

    model = CRNNWordEncoder(num_classes, hidden_size, lstm_layers, rnn_dropout, lstm_dropout, pretrained_backbone).to(env.device)


    loss_func = CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    
    # Label Smoothing для борьбы с переобучением
    label_smoothing = 0.1  # 10% smoothing
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)

    steps_per_epoch = len(train_loader)
    lr_sched = OneCycleLR(
        opt,
        max_lr=5e-4,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 30% эпох на разогрев
        anneal_strategy='cos'
    )
    # lr_sched = ReduceLROnPlateau(
    #     opt,
    #     'min',
    #     factor=0.5,
    #     patience=5,
    #     threshold=0.01,
    #     threshold_mode='rel',
    #     min_lr=1e-6
    # )

    early_stopping_mode = False
    threshold_loss = 0.01
    threshold_metric = 0.02
    early_stopping = 15
    models_dir = WORKDIR / 'ml' / 'word_decoder' / 'model_weights' / f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    models_dir.mkdir(parents=True)

# ======================================================================================================================
# Обучение
# ======================================================================================================================

    log_event(f'\033[34mОбучение началось\033[0m | Эпохи: \033[33m{epochs}\033[0m')

    train_loss_list, val_loss_list, lr_list, cer_list, wer_list, min_loss, min_metric_value = [], [], [], [], [], None, None
    plateau_loss_epochs = 0


    for epoch in range(1, epochs + 1):

        model.train()

        list_train_loss = []
        train_loop = tqdm(train_loader, leave=False, desc=f'Training \033[34m#{epoch}\033[0m')
        
        opt.zero_grad()  # Один раз в начале эпохи
        
        for i, (images, targets, _, target_lengths) in enumerate(train_loop):
            images = images.to(env.device, non_blocking=True)
            targets = targets.to(env.device, non_blocking=True)
            target_lengths = target_lengths.to(env.device, non_blocking=True)
            
            "Forward"
            log_probs = model(images)  # [seq_len, batch, num_classes]
            log_probs = torch.nn.functional.log_softmax(log_probs, dim=2)
            
            # Вычисляем input_lengths (длина последовательности после модели)
            # Для нашей модели: seq_len из выхода модели
            batch_size = images.shape[0]
            input_lengths = torch.full((batch_size,), log_probs.shape[0], dtype=torch.long, device=env.device)
            
            "Loss"
            loss = loss_func(log_probs, targets, input_lengths, target_lengths)
            
            # Label Smoothing: добавляем небольшой uniform loss
            if label_smoothing > 0:
                # Uniform distribution loss
                uniform_loss = -log_probs.mean()
                loss = (1 - label_smoothing) * loss + label_smoothing * uniform_loss
            
            "Backward"
            if gradient_accumulation_mode:
                loss = loss / accumulation_steps
            
            loss.backward()
            
            "Gradient clipping для стабильности"
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            "Gradient Accumulation"
            if gradient_accumulation_mode and (i + 1) % accumulation_steps == 0:
                opt.step()
                lr_sched.step()  # OneCycleLR обновляется каждый батч!
                opt.zero_grad()
            elif not gradient_accumulation_mode:
                opt.step()
                lr_sched.step()  # OneCycleLR обновляется каждый батч!
                opt.zero_grad()
            
            "Считаем loss"
            batch_loss = loss.item() * (accumulation_steps if gradient_accumulation_mode else 1)
            list_train_loss.append(batch_loss)

            train_loop.set_postfix({'loss': f'\033[32m{batch_loss:.4f}\033[0m'})
        
        # Если последний батч не кратен accumulation_steps
        if gradient_accumulation_mode and (i + 1) % accumulation_steps != 0:
            opt.step()
            lr_sched.step()
            opt.zero_grad()
        avg_train_loss = sum(list_train_loss) / len(list_train_loss)
        train_loss_list.append(avg_train_loss)
        
        log_event(f"\033[32mTRAINING\033[0m | Epoch {epoch} | train_loss=\033[33m{avg_train_loss:.4f}\033[0m")

    # ==================================================================================================================
    # Валидация
    # ==================================================================================================================

        model.eval()
        with torch.no_grad():
            all_predictions = []
            all_targets = []

            list_val_loss = []
            val_loop = tqdm(val_loader, leave=False, desc=f'Validation \033[36m#{epoch}\033[0m')
            for images, targets, _, target_lengths in val_loop:

                images = images.to(env.device, non_blocking=True)
                targets_gpu = targets.to(env.device, non_blocking=True)
                target_lengths_gpu = target_lengths.to(env.device, non_blocking=True)

                "Forward"
                log_probs = model(images)  # [seq_len, batch, num_classes]
                log_probs_softmax = torch.nn.functional.log_softmax(log_probs, dim=2)

                # Вычисляем input_lengths
                batch_size = images.shape[0]
                input_lengths = torch.full((batch_size,), log_probs.shape[0], dtype=torch.long, device=env.device)

                "Loss"
                loss = loss_func(log_probs_softmax, targets_gpu, input_lengths, target_lengths_gpu)

                list_val_loss.append(loss.item())

                "Декодируем предсказания для метрик"
                predictions = decode_predictions(log_probs, val_dset, blank_idx=0)
                all_predictions.extend(predictions)

                "Декодируем целевые тексты"
                targets_cpu = targets.cpu()
                target_lengths_cpu = target_lengths.cpu()
                start_idx = 0
                for length in target_lengths_cpu:
                    target_indices = targets_cpu[start_idx:start_idx + length].tolist()
                    target_text = val_dset.indices_to_text(target_indices)
                    all_targets.append(target_text)
                    start_idx += length

                val_loop.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = sum(list_val_loss) / len(list_val_loss)
        val_loss_list.append(avg_val_loss)

        "Метрики"
        cer = calculate_cer(all_predictions, all_targets)
        wer = calculate_wer(all_predictions, all_targets)
        avg_metric_value = (cer + wer) / 2.0
        
        cer_list.append(cer)
        wer_list.append(wer)

        "ReduceOnPlateau"
        # lr_sched.step(avg_val_loss)
        # lr = lr_sched.get_last_lr()[0]

        "OneCycle Scheduler"
        lr = opt.param_groups[0]['lr']
        lr_list.append(lr)

        log_event(f"\033[34mVALIDATION\033[0m Epoch {epoch} | val_loss=\033[35m{avg_val_loss:.4f}\033[0m | CER=\033[32m{cer:.2f}%\033[0m | WER=\033[36m{wer:.2f}%\033[0m | LR=\033[33m{lr:.6f}\033[0m")

        history = {
            "general_metrics": {
                'val_loss_list': val_loss_list,
                'train_loss_list': train_loss_list,
                'cer_list': cer_list,
                'wer_list': wer_list,
            },
            "train_loss_last": avg_train_loss,
            "val_loss_last": avg_val_loss,
            "cer_cur": cer,
            "wer_cur": wer,
            "lr": lr_list
        }

        "Сохраняем модель, как только виднеется прогресс"
        min_loss = avg_val_loss if min_loss is None else min_loss
        min_metric_value = avg_metric_value if min_metric_value is None else min_metric_value

        "Сохраняем, если модель стала лучше на 1% по метрике ИЛИ лоссу за трейн ран"
        if (min_loss - min_loss * threshold_loss > avg_val_loss) or (min_metric_value - min_metric_value * threshold_metric > avg_metric_value):
            min_loss = avg_val_loss
            min_metric_value = avg_metric_value

            checkpoint = {
                'model_code': model_word_encoder_code,
                'model_params': {
                    'num_classes': num_classes,
                    'hidden_size': hidden_size,
                    'num_lstm_layers': lstm_layers,
                    'rnn_dropout': rnn_dropout,
                    'lstm_dropout': lstm_dropout
                },
                'state_model': model.state_dict(),
                'state_opt': opt.state_dict(),
                'state_lr_scheduler': lr_sched.state_dict(),
                'save_epoch': epoch,
                'history': history,
                'charset': train_dset.charset
            }
            torch.save(checkpoint, models_dir.joinpath(f'model_epoch{epoch}.pth'))
            plateau_loss_epochs = 0
            log_event(f'На эпохе - \033[35m{epoch}\033[0m модель сохранена | val_loss=\033[34m{avg_val_loss:.4f}\033[0m, CER=\033[32m{cer:.2f}\033[0m%, WER=\033[34m{wer:.2f}\033[0m%', level='WARNING')

        "Early Stopping"
        if early_stopping_mode and plateau_loss_epochs >= early_stopping:
            log_event(f'\033[31m{'!!!' * 10} Принудительная остановка обучения, нет прогресса {'!!!' * 10}\033[0m', level='WARNING')
            raise Exception("Early Stopping")

        plateau_loss_epochs += 1

    plot_loss_dynamics(history, models_dir / 'loss_distribution.png')
    plot_metrics_dynamics(history, models_dir / 'metrics.png')
    plot_lr_chronology(history, models_dir / 'chronology.png')

    log_event(f'\033[34m{'>>>' * 10} Обучение завершено {'<<<' * 10}\033[0m')

if __name__ == '__main__':
    train_run()
