from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.config import env, WORKDIR
from ml.logger_config import log_event
from ml.word_decoder.dataset_class.beam_search_decoder import BeamSearchDecoder
from ml.word_decoder.dataset_class.dataclass_word_decoder import CRNNWordDataset
from ml.word_decoder.metrics import decode_predictions, calculate_cer, calculate_wer, calculate_accuracy
from ml.word_decoder.models import CRNNWordDecoder
from ml.word_decoder.utils import parse_args


def test_run(weights_path: Path, batch_size: int = 64, img_height: int = 64, workers: int = 2):
    """"""

    batch_size_test = batch_size
    dataload_workers = workers
    prefetch_factor = 2

    base_dset_path = WORKDIR / 'dataset' / 'iam-words'
    test_dset = CRNNWordDataset(base_dset_path / 'test', base_dset_path / 'charset.txt', img_height, 'test')
    test_loader = DataLoader(
        dataset=test_dset,
        batch_size=batch_size_test,
        num_workers=dataload_workers,
        collate_fn=CRNNWordDataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )


    "Выгружаем Параметры модели"
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл с весами не найден: {weights_path}")

    log_event(f'Загрузка весов из: \033[36m{weights_path}\033[0m', level='WARNING')

    model_params = torch.load(weights_path, weights_only=False, map_location=env.device)

    model_inner_params = model_params['model_params']
    beam_size = model_params.get('beam_search_decoder_size', 10)
    hidden_size, num_lstm_layers, lstm_dropout = model_inner_params['hidden_size'], model_inner_params['num_lstm_layers'], model_inner_params['lstm_dropout']
    charset, use_feature_compressor = model_params['charset'], model_inner_params.get('use_feature_compressor', False),

    model = CRNNWordDecoder(
        len(charset),
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        lstm_dropout=lstm_dropout,
        use_feature_compressor=use_feature_compressor,
    )
    model.to(env.device)

    "Загружаем веса"
    model_weights = model_params['state_model']
    model.load_state_dict(model_weights)

    log_event(f'✅ Веса загружены успешно', level='WARNING')

    "Некоторые гиперпараметры"
    loss_func = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    beam_search_decoder = BeamSearchDecoder(
        tokens=charset,
        beam_size=beam_size,
        nbest=1,
        use_cuda=True
    )
    use_spell_checker = False
    spell_checker = SpellChecker(
        vocabulary_path=WORKDIR / 'ml' / 'word_decoder' / 'model_weights' / 'vocabulary.json',
        max_edit_distance=2,
        min_word_length=3
    )

    log_event(f'Начали Тестирование модели. Spell Checker status: \033[31m{use_spell_checker}\033[0m')
    model.eval()
    with torch.no_grad():
        list_test_loss = []

        all_predictions = []
        all_targets = []

        test_loop = tqdm(test_loader, leave=False, desc=f'Testing')
        for images, targets, images_widths, target_lengths in test_loop:

            images = images.to(env.device, non_blocking=True)
            targets_gpu = targets.to(env.device, non_blocking=True)
            images_widths = images_widths.to(env.device, non_blocking=True)
            target_lengths_gpu = target_lengths.to(env.device, non_blocking=True)

            "Forward"
            log_probs = model(images)  # [seq_len, batch, num_classes]
            log_probs_softmax = torch.nn.functional.log_softmax(log_probs, dim=2)

            input_lengths = torch.clamp(images_widths, min=target_lengths.max().item())

            "Loss"
            loss = loss_func(log_probs_softmax, targets_gpu, input_lengths, target_lengths_gpu)

            list_test_loss.append(loss.item())
            log_probs_for_beam = log_probs_softmax.transpose(0, 1).contiguous()

            "Декодируем предсказания для метрик"
            predictions = beam_search_decoder.decode(log_probs_for_beam, lengths=input_lengths)
            all_predictions.extend(predictions)

            "Декодируем целевые тексты"
            targets_cpu = targets.cpu()
            target_lengths_cpu = target_lengths.cpu()
            start_idx = 0
            for length in target_lengths_cpu:
                target_indices = targets_cpu[start_idx:start_idx + length].tolist()
                target_text = test_dset.indices_to_text(target_indices)
                all_targets.append(target_text)
                start_idx += length

            test_loop.set_postfix({'loss': f'{loss.item():.4f}'})


    "Метрики"
    if use_spell_checker:
        all_predictions = spell_checker.correct_text(all_predictions)
    cer = calculate_cer(all_predictions, all_targets)
    wer = calculate_wer(all_predictions, all_targets)
    acc = calculate_accuracy(all_predictions, all_targets)
    avg_test_loss = sum(list_test_loss) / len(list_test_loss)

    log_event(f"\033[35mTESTING\033[0m | test_loss=\033[35m{avg_test_loss:.4f}\033[0m | CER=\033[32m{cer:.2f}%\033[0m | WER=\033[36m{wer:.2f}%\033[0m | ACC=\033[35m{acc:.2f}%\033[0m", level='WARNING')



if __name__ == '__main__':
    args = parse_args()

    weights_path_arg = Path(args.weights)

    # Если путь относительный, делаем его относительно WORKDIR
    if not weights_path_arg.is_absolute():
        weights_path_arg = WORKDIR / weights_path_arg

    # Запуск
    test_run(
        weights_path=weights_path_arg,
        batch_size=args.batch_size,
        img_height=args.img_height,
        workers=args.workers
    )
