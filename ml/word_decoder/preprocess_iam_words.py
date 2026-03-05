"""
Препроцессинг IAM датасета: вырезка отдельных слов из полных изображений форм.

Структура входных данных:
    dataset/iam-form-stratified/
    ├── train/
    │   ├── imgs/
    │   │   └── a01-000u.png
    │   └── annotations/
    │       └── a01-000u.xml
    └── test/
        └── ...

Структура выходных данных:
    dataset/iam-words/
    ├── train/
    │   ├── imgs/
    │   │   ├── a01-000u_a_0.png
    │   │   ├── a01-000u_move_1.png
    │   │   └── ...
    │   └── annotations/
    │       ├── a01-000u_a_0.txt  (содержит "a")
    │       ├── a01-000u_move_1.txt  (содержит "move")
    │       └── ...
    └── test/
        └── ...
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal
from collections import Counter

from PIL import Image
from tqdm import tqdm


# Финальный алфавит (43 символа)
CHARSET = [
    '<blank>',  # 0 - для CTC
    # 26 букв (lowercase)
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    # 10 цифр
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    # 6 спец символов
    ' ', '.', '!', '?', ':', '-'
]

CHARSET_SET = set(CHARSET[1:])  # без <blank>

# Маппинг спец символов для безопасных имён файлов
CHAR_TO_SAFE_FILENAME = {
    ' ': '_space_',
    '.': '_dot_',
    '!': '_excl_',
    '?': '_quest_',
    ':': '_colon_',
    '-': '_dash_'
}


def text_to_safe_filename(text: str, max_length: int = 50) -> str:
    """
    Преобразует текст в безопасное имя файла, заменяя спец символы.
    
    Args:
        text: текст (уже в lowercase)
        max_length: максимальная длина результата
        
    Returns:
        безопасное имя для файла
    """
    safe_text = text
    for char, replacement in CHAR_TO_SAFE_FILENAME.items():
        safe_text = safe_text.replace(char, replacement)
    
    # Ограничиваем длину
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length]
    
    return safe_text


def is_valid_text(text: str) -> bool:
    """
    Проверяет, содержит ли текст только допустимые символы из алфавита.
    
    Args:
        text: текст для проверки
        
    Returns:
        True если все символы из алфавита, False иначе
    """
    text_lower = text.lower()
    return all(char in CHARSET_SET for char in text_lower)


def extract_word_bbox(word_element) -> tuple[int, int, int, int] | None:
    """
    Извлекает bounding box слова из XML элемента.
    
    Args:
        word_element: XML элемент <word>
        
    Returns:
        (x1, y1, x2, y2) или None если bbox некорректный
    """
    cmps = word_element.findall('cmp')
    if len(cmps) == 0:
        return None
    
    xs, ys, xe, ye = [], [], [], []
    
    for cmp in cmps:
        try:
            x = int(cmp.attrib['x'])
            y = int(cmp.attrib['y'])
            w = int(cmp.attrib['width'])
            h = int(cmp.attrib['height'])
            
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)
        except (KeyError, ValueError):
            continue
    
    if not xs:
        return None
    
    # Левый верхний и правый нижний углы
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xe), max(ye)
    
    # Проверка корректности
    if x2 <= x1 or y2 <= y1:
        return None
    
    return x1, y1, x2, y2


def process_form(
    img_path: Path,
    ann_path: Path,
    output_imgs_dir: Path,
    output_anns_dir: Path,
    word_counter: Counter
) -> tuple[int, int, int]:
    """
    Обрабатывает одну форму: вырезает слова и сохраняет их.
    
    Args:
        img_path: путь к изображению формы
        ann_path: путь к XML аннотации
        output_imgs_dir: директория для сохранения изображений слов
        output_anns_dir: директория для сохранения текстовых аннотаций
        word_counter: счётчик для отслеживания повторяющихся слов
        
    Returns:
        (total_words, saved_words, skipped_words)
    """
    # Парсим XML
    try:
        tree = ET.parse(ann_path)
        root = tree.getroot()
    except ET.ParseError:
        return 0, 0, 0
    
    # Загружаем изображение
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return 0, 0, 0
    
    form_id = img_path.stem  # например, "a01-000u"
    
    total_words = 0
    saved_words = 0
    skipped_words = 0
    
    # Проходим по всем словам в форме
    for line in root.findall('.//line'):
        for word in line.findall('word'):
            total_words += 1
            
            # Извлекаем текст
            text = word.attrib.get('text', '').strip()
            if not text:
                skipped_words += 1
                continue
            
            # Проверяем, что все символы из нашего алфавита
            if not is_valid_text(text):
                skipped_words += 1
                continue
            
            # Извлекаем bbox
            bbox = extract_word_bbox(word)
            if bbox is None:
                skipped_words += 1
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Вырезаем слово из изображения
            try:
                word_img = img.crop((x1, y1, x2, y2))
            except Exception:
                skipped_words += 1
                continue
            
            # Формируем имя файла: form_id_word_count
            # Например: a01-000u_hello_0.png или a01-000u__quest__0.png для "?"
            text_lower = text.lower()
            safe_text = text_to_safe_filename(text_lower)
            
            word_count = word_counter[f"{form_id}_{safe_text}"]
            word_counter[f"{form_id}_{safe_text}"] += 1
            
            filename = f"{form_id}_{safe_text}_{word_count}"
            
            # Сохраняем изображение
            img_output_path = output_imgs_dir / f"{filename}.png"
            word_img.save(img_output_path)
            
            # Сохраняем текст (в lowercase)
            ann_output_path = output_anns_dir / f"{filename}.txt"
            with open(ann_output_path, 'w', encoding='utf-8') as f:
                f.write(text_lower)
            
            saved_words += 1
    
    return total_words, saved_words, skipped_words


def preprocess_split(
    input_dir: Path,
    output_dir: Path,
    split: Literal['train', 'test']
):
    """
    Обрабатывает один split (train или test).
    
    Args:
        input_dir: директория с исходными данными
        output_dir: директория для сохранения результатов
        split: 'train' или 'test'
    """
    print(f"\n{'='*60}")
    print(f"Обработка {split} split")
    print(f"{'='*60}")
    
    # Пути к входным данным
    input_imgs_dir = input_dir / split / 'imgs'
    input_anns_dir = input_dir / split / 'annotations'
    
    # Пути к выходным данным
    output_imgs_dir = output_dir / split / 'imgs'
    output_anns_dir = output_dir / split / 'annotations'
    
    # Создаём директории
    output_imgs_dir.mkdir(parents=True, exist_ok=True)
    output_anns_dir.mkdir(parents=True, exist_ok=True)
    
    # Получаем список всех изображений
    img_files = sorted(input_imgs_dir.glob('*.png'))
    
    if not img_files:
        print(f"⚠️  Не найдено изображений в {input_imgs_dir}")
        return
    
    print(f"Найдено {len(img_files)} форм")
    
    # Счётчики
    word_counter = Counter()
    total_forms = 0
    total_words_all = 0
    saved_words_all = 0
    skipped_words_all = 0
    
    # Обрабатываем каждую форму
    for img_path in tqdm(img_files, desc=f"Обработка {split}"):
        ann_path = input_anns_dir / f"{img_path.stem}.xml"
        
        if not ann_path.exists():
            continue
        
        total_forms += 1
        
        total_words, saved_words, skipped_words = process_form(
            img_path=img_path,
            ann_path=ann_path,
            output_imgs_dir=output_imgs_dir,
            output_anns_dir=output_anns_dir,
            word_counter=word_counter
        )
        
        total_words_all += total_words
        saved_words_all += saved_words
        skipped_words_all += skipped_words
    
    # Статистика
    print(f"\n{'='*60}")
    print(f"Статистика для {split}:")
    print(f"{'='*60}")
    print(f"Обработано форм: {total_forms}")
    print(f"Всего слов найдено: {total_words_all}")
    print(f"Сохранено слов: {saved_words_all}")
    print(f"Пропущено слов: {skipped_words_all}")
    
    if total_words_all > 0:
        success_rate = (saved_words_all / total_words_all) * 100
        print(f"Процент успеха: {success_rate:.1f}%")


def main():
    """Главная функция препроцессинга."""
    # Пути
    input_dir = Path('dataset/iam-form-stratified')
    output_dir = Path('dataset/iam-words')
    
    print("="*60)
    print("Препроцессинг IAM датасета для CRNN")
    print("="*60)
    print(f"Входная директория: {input_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Алфавит: {len(CHARSET)} символов")
    print(f"  - 1 blank символ")
    print(f"  - 26 букв (lowercase)")
    print(f"  - 10 цифр")
    print(f"  - 6 спец символов: {CHARSET[-6:]}")
    
    # Проверяем существование входной директории
    if not input_dir.exists():
        print(f"\n❌ Ошибка: директория {input_dir} не существует!")
        return
    
    # Обрабатываем train и test
    for split in ['test']:
        split_dir = input_dir / split
        if split_dir.exists():
            preprocess_split(input_dir, output_dir, split)
        else:
            print(f"\n⚠️  Пропускаем {split}: директория не найдена")
    
    print("\n" + "="*60)
    print("✅ Препроцессинг завершён!")
    print("="*60)
    print(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
