"""
Метрики для оценки качества OCR модели.

CER (Character Error Rate) - процент ошибок на уровне символов
WER (Word Error Rate) - процент ошибок на уровне слов
"""

import torch
import Levenshtein


def calculate_cer(predictions: list[str], targets: list[str]) -> float:
    """
    Вычисляет Character Error Rate (CER).
    
    CER = (Substitutions + Deletions + Insertions) / Total Characters
    
    Args:
        predictions: список предсказанных текстов
        targets: список целевых текстов
        
    Returns:
        CER в процентах (0-100)
    """
    if len(predictions) != len(targets):
        raise ValueError("Количество предсказаний и целей должно совпадать")
    
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predictions, targets):
        # Расстояние Левенштейна (количество операций для преобразования)
        distance = Levenshtein.distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    if total_length == 0:
        return 0.0
    
    cer = (total_distance / total_length) * 100
    return cer


def calculate_wer(predictions: list[str], targets: list[str]) -> float:
    """
    Вычисляет Word Error Rate (WER).
    
    WER = (Substitutions + Deletions + Insertions) / Total Words
    
    Args:
        predictions: список предсказанных текстов
        targets: список целевых текстов
        
    Returns:
        WER в процентах (0-100)
    """
    if len(predictions) != len(targets):
        raise ValueError("Количество предсказаний и целей должно совпадать")
    
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Разбиваем на слова
        pred_words = pred.split()
        target_words = target.split()
        
        # Расстояние Левенштейна на уровне слов
        distance = Levenshtein.distance(pred_words, target_words)
        total_distance += distance
        total_words += len(target_words)
    
    if total_words == 0:
        return 0.0
    
    wer = (total_distance / total_words) * 100
    return wer


def ctc_greedy_decode(predictions: torch.Tensor, blank_idx: int = 0) -> list[list[int]]:
    """
    Простой greedy декодер для CTC.
    
    Args:
        predictions: [seq_len, batch, num_classes] или [batch, seq_len, num_classes]
        blank_idx: индекс blank символа
        
    Returns:
        список списков индексов (декодированные последовательности)
    """
    # Если формат [seq_len, batch, num_classes], транспонируем
    if predictions.dim() == 3 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.permute(1, 0, 2)  # [batch, seq_len, num_classes]
    
    # Берём argmax
    _, preds = predictions.max(2)  # [batch, seq_len]
    
    decoded = []
    for pred in preds:
        # Убираем повторяющиеся символы и blank
        pred_list = []
        prev_char = None
        for char_idx in pred.tolist():
            if char_idx != blank_idx and char_idx != prev_char:
                pred_list.append(char_idx)
            prev_char = char_idx
        decoded.append(pred_list)
    
    return decoded


def decode_predictions(
    predictions: torch.Tensor,
    dataset,
    blank_idx: int = 0
) -> list[str]:
    """
    Декодирует предсказания модели в текст.
    
    Args:
        predictions: [seq_len, batch, num_classes] - выход модели
        dataset: CRNNWordDataset для конвертации индексов в текст
        blank_idx: индекс blank символа
        
    Returns:
        список строк (декодированные тексты)
    """
    # Greedy декодирование
    decoded_indices = ctc_greedy_decode(predictions, blank_idx)
    
    # Конвертируем индексы в текст
    decoded_texts = [dataset.indices_to_text(indices) for indices in decoded_indices]
    
    return decoded_texts


def calculate_accuracy(predictions: list[str], targets: list[str]) -> float:
    """
    Вычисляет точность на уровне полного совпадения слов.
    
    Args:
        predictions: список предсказанных текстов
        targets: список целевых текстов
        
    Returns:
        точность в процентах (0-100)
    """
    if len(predictions) != len(targets):
        raise ValueError("Количество предсказаний и целей должно совпадать")
    
    correct = sum(pred == target for pred, target in zip(predictions, targets))
    accuracy = (correct / len(predictions)) * 100
    
    return accuracy


# Пример использования
if __name__ == "__main__":
    # Тестовые данные
    predictions = ["hello", "wrld", "test", "ocr"]
    targets = ["hello", "world", "test", "ocr"]
    
    cer = calculate_cer(predictions, targets)
    wer = calculate_wer(predictions, targets)
    acc = calculate_accuracy(predictions, targets)
    
    print(f"CER: {cer:.2f}%")
    print(f"WER: {wer:.2f}%")
    print(f"Accuracy: {acc:.2f}%")
    
    # Детальный анализ
    print("\nДетальный анализ:")
    for pred, target in zip(predictions, targets):
        dist = Levenshtein.distance(pred, target)
        print(f"  '{target}' -> '{pred}' | distance: {dist}")
