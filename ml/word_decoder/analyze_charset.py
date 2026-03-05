"""
Анализ частоты символов в препроцессированном IAM датасете.
"""

from pathlib import Path
from collections import Counter
from tqdm import tqdm


def analyze_charset(dataset_path: Path):
    """
    Анализирует частоту символов во всех splits датасета.
    
    Args:
        dataset_path: путь к препроцессированному датасету
    """
    print("="*60)
    print("Анализ частоты символов в IAM датасете")
    print("="*60)
    
    all_chars = Counter()
    total_words = 0
    total_chars = 0
    
    # Проходим по всем splits
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split / 'annotations'
        
        if not split_path.exists():
            print(f"⚠️  Пропускаем {split}: директория не найдена")
            continue
        
        print(f"\nОбработка {split}...")
        
        txt_files = list(split_path.glob('*.txt'))
        split_chars = Counter()
        split_words = 0
        
        for txt_file in tqdm(txt_files, desc=f"Анализ {split}"):
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                
                if text:
                    split_words += 1
                    split_chars.update(text)
        
        all_chars.update(split_chars)
        total_words += split_words
        
        split_total_chars = sum(split_chars.values())
        total_chars += split_total_chars
        
        print(f"  Слов: {split_words:,}")
        print(f"  Символов: {split_total_chars:,}")
        print(f"  Уникальных символов: {len(split_chars)}")
    
    # Общая статистика
    print("\n" + "="*60)
    print("ОБЩАЯ СТАТИСТИКА")
    print("="*60)
    print(f"Всего слов: {total_words:,}")
    print(f"Всего символов: {total_chars:,}")
    print(f"Уникальных символов: {len(all_chars)}")
    print(f"Средняя длина слова: {total_chars / total_words:.2f} символов")
    
    # Рейтинг символов
    print("\n" + "="*60)
    print("РЕЙТИНГ СИМВОЛОВ (по частоте)")
    print("="*60)
    print(f"{'Ранг':<6} {'Символ':<10} {'Количество':<15} {'Процент':<10} {'Накопленный %'}")
    print("-"*60)
    
    cumulative_percent = 0
    for rank, (char, count) in enumerate(all_chars.most_common(), start=1):
        percent = (count / total_chars) * 100
        cumulative_percent += percent
        
        # Отображаем спец символы читаемо
        if char == ' ':
            char_display = '<space>'
        elif char == '\n':
            char_display = '<newline>'
        elif char == '\t':
            char_display = '<tab>'
        else:
            char_display = char
        
        print(f"{rank:<6} {char_display:<10} {count:<15,} {percent:<10.2f} {cumulative_percent:.2f}%")
    
    # Категории символов
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО КАТЕГОРИЯМ")
    print("="*60)
    
    letters = sum(count for char, count in all_chars.items() if char.isalpha())
    digits = sum(count for char, count in all_chars.items() if char.isdigit())
    spaces = all_chars.get(' ', 0)
    punctuation = sum(count for char, count in all_chars.items() 
                     if char in '.!?:,-;\'\"()[]{}')
    other = total_chars - letters - digits - spaces - punctuation
    
    print(f"Буквы:       {letters:>10,} ({letters/total_chars*100:>5.1f}%)")
    print(f"Цифры:       {digits:>10,} ({digits/total_chars*100:>5.1f}%)")
    print(f"Пробелы:     {spaces:>10,} ({spaces/total_chars*100:>5.1f}%)")
    print(f"Пунктуация:  {punctuation:>10,} ({punctuation/total_chars*100:>5.1f}%)")
    print(f"Другое:      {other:>10,} ({other/total_chars*100:>5.1f}%)")
    
    # Топ-10 букв
    print("\n" + "="*60)
    print("ТОП-10 БУКВ")
    print("="*60)
    
    letter_counts = Counter({char: count for char, count in all_chars.items() if char.isalpha()})
    for rank, (char, count) in enumerate(letter_counts.most_common(10), start=1):
        percent = (count / letters) * 100
        print(f"{rank:>2}. {char} - {count:>8,} ({percent:>5.2f}% от всех букв)")
    
    # Топ-10 слов
    print("\n" + "="*60)
    print("ТОП-20 САМЫХ ЧАСТЫХ СЛОВ")
    print("="*60)
    
    word_counts = Counter()
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split / 'annotations'
        if split_path.exists():
            for txt_file in split_path.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    word = f.read().strip()
                    if word:
                        word_counts[word] += 1
    
    for rank, (word, count) in enumerate(word_counts.most_common(20), start=1):
        percent = (count / total_words) * 100
        print(f"{rank:>2}. '{word}' - {count:>6,} раз ({percent:>5.2f}%)")
    
    # Распределение длин слов
    print("\n" + "="*60)
    print("РАСПРЕДЕЛЕНИЕ ДЛИН СЛОВ")
    print("="*60)
    
    word_lengths = Counter()
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split / 'annotations'
        if split_path.exists():
            for txt_file in split_path.glob('*.txt'):
                with open(txt_file, 'r', encoding='utf-8') as f:
                    word = f.read().strip()
                    if word:
                        word_lengths[len(word)] += 1
    
    print(f"{'Длина':<10} {'Количество':<15} {'Процент':<10} {'Гистограмма'}")
    print("-"*60)
    
    max_count = max(word_lengths.values())
    for length in sorted(word_lengths.keys()):
        count = word_lengths[length]
        percent = (count / total_words) * 100
        bar_length = int((count / max_count) * 40)
        bar = '█' * bar_length
        print(f"{length:<10} {count:<15,} {percent:<10.2f} {bar}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    dataset_path = Path('dataset/iam-words')
    
    if not dataset_path.exists():
        print(f"❌ Ошибка: датасет не найден в {dataset_path}")
        print("Сначала запустите preprocess_iam_words.py")
    else:
        analyze_charset(dataset_path)
