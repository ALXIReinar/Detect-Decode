"""
Spell Checker для OCR - коррекция распознанных слов без KenLM.

Подход:
1. Словарь частотных слов из тренировочного набора
2. Levenshtein distance для поиска ближайшего слова
3. Коррекция только слов с низкой уверенностью или не из словаря
"""
from pathlib import Path
from typing import Optional
import json
from collections import Counter, defaultdict

import Levenshtein
import numpy as np

from ml.logger_config import log_event


class SpellChecker:
    """
    Spell checker для OCR на основе словаря и Levenshtein distance.
    
    Алгоритм:
    1. Если слово в словаре → отдаём как есть
    2. Если слово не в словаре → ищем ближайшее
    3. Если расстояние <= max_distance → заменяем
    """
    
    def __init__(
        self,
        vocabulary_path: Optional[Path | str] = None,
        max_edit_distance: int = 2,
        min_word_length: int = 3
    ):
        """
        Args:
            vocabulary_path: путь к JSON файлу со словарём {word: frequency}
            max_edit_distance: максимальное расстояние Левенштейна (default: 2)
            min_word_length: минимальная длина слова для коррекции (default: 3)
        """
        self.max_edit_distance = max_edit_distance
        self.min_word_length = min_word_length
        
        "Загружаем словарь"
        if vocabulary_path is not None:
            self.vocabulary = self.load_vocabulary(vocabulary_path)
        else:
            self.vocabulary = {}

        "Собираем словарь 'длина_слова: список_слов_с_одной_длиной'"
        self.words_by_len = defaultdict(list)
        for word in self.vocabulary.keys():
            self.words_by_len[len(word)].append(word) # Нужно для суженого перебора слов при поиске исправления


    
    def load_vocabulary(self, path: Path | str) -> dict[str, int]:
        """
        Загружает словарь из JSON файла.
        
        Args:
            path: путь к JSON файлу
            
        Returns:
            словарь {word: frequency}
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            vocabulary = json.load(f)
        
        return vocabulary


    def save_vocabulary(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, ensure_ascii=False, indent=2)


    def build_vocabulary_from_texts(self, texts: list[str], min_frequency: int = 2):
        """
        Строит словарь из списка текстов.
        
        Args:
            texts: список текстов
            min_frequency: минимальная частота слова для включения в словарь
        """
        word_counter = Counter()
        
        for text in texts:
            # Нормализуем текст
            text_normalized = text.lower().strip()
            words = text_normalized.split()
            
            # Фильтруем короткие слова и не-буквенные
            words_filtered = [
                w for w in words 
                if len(w) >= self.min_word_length and w.isalpha()
            ]
            
            word_counter.update(words_filtered)
        
        # Фильтруем по частоте
        self.vocabulary = {
            word: count 
            for word, count in word_counter.items() 
            if count >= min_frequency
        }
        
        return self.vocabulary


    def find_closest_word(self, word: str) -> Optional[str]:
        word_lower = word.lower()

        "Если слово в словаре => возвращаем его"
        if word_lower in self.vocabulary:
            return word_lower

        target_len = len(word_lower)
        candidates = []

        "Итерируемся по диапазону допустимой разницы в длине слов"
        for l in range(target_len - self.max_edit_distance, target_len + self.max_edit_distance + 1):

            "Прогоняем каждое слово из словаря + score"
            for vocab_word in self.words_by_len.get(l, []):
                dist = Levenshtein.distance(word_lower, vocab_word)
                if dist <= self.max_edit_distance:

                    # Логарифмический "рейтинг"
                    freq = self.vocabulary[vocab_word]
                    score = dist * 2 - np.log1p(freq)
                    candidates.append((vocab_word, score))

        "Не нашли исправление"
        if not candidates: return None

        "Выбираем самое лучшее исправление по score"
        return min(candidates, key=lambda x: x[1])[0]


    def correct_word(self, word: str) -> str:
        # Пропускаем короткие слова
        if len(word) < self.min_word_length:
            return word
        
        # Пропускаем не-буквенные слова
        if not word.isalpha():
            return word

        # Ищем коррекцию
        corrected = self.find_closest_word(word)
        
        if corrected is not None:
            return corrected
        
        return word
    
    def correct_text(self, words: list[str] | str) -> list[str]:
        if isinstance(words, str):
            words = words.split()

        "Получаем исправленные слова"
        return [self.correct_word(word) for word in words]


def build_vocabulary_from_dataset(
    dataset_path: Path | str,
    output_path: Path | str,
    min_frequency: int = 2
):
    """
    Строит словарь из датасета IAM Words.
    
    Args:
        dataset_path: путь к датасету (train/)
        output_path: путь для сохранения словаря
        min_frequency: минимальная частота слова
    """
    dataset_path = Path(dataset_path)
    
    # Читаем все тексты из датасета
    texts = []
    
    # Ищем все .txt файлы с ground truth
    for txt_file in dataset_path.rglob('*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            if text:
                texts.append(text)
    
    # Строим словарь
    spell_checker = SpellChecker()
    vocabulary = spell_checker.build_vocabulary_from_texts(texts, min_frequency)
    

    spell_checker.save_vocabulary(output_path)
    log_event(f"Сформирован словарь из слов: \033[32m{len(vocabulary)}\033[0m | Сохранён в \033[34m{output_path}\033[0m", level="WARNING")
    return vocabulary



if __name__ == '__main__':
    from ml.config import WORKDIR

    # Пример: строим словарь из тренировочного набора
    dataset_path = WORKDIR / 'dataset' / 'iam-words' / 'train'
    output_path = WORKDIR / 'ml' / 'word_decoder' / 'model_weights' / 'vocabulary.json'
    
    vocabulary = build_vocabulary_from_dataset(
        dataset_path,
        output_path,
        min_frequency=2
    )
