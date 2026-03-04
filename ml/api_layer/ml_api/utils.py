"""
Утилиты для обработки изображений и создания архивов.
"""
import shutil
import tarfile
from pathlib import Path

from PIL import Image
from fastapi import UploadFile

from ml.config import S3_OCR_DIR, S3_TEMP_DIR
from ml.logger_config import log_event


async def save_uploaded_image(file: UploadFile, img_id: int) -> Path:
    """"""
    "Директория для изображения"
    img_dir = S3_OCR_DIR / str(img_id)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    "Сохраняем само изображение"
    img_path = img_dir / 'original.png'
    content = await file.read()
    with open(img_path, 'wb') as f:
        f.write(content)
    
    log_event(f"Сохранили image {img_id} to {img_path}", level='INFO')
    return img_path


def load_image_from_path(img_path: Path) -> Image.Image:
    img = Image.open(img_path).convert('RGB')
    return img


def save_word_crop(word_img: Image.Image, img_id: int, word_idx: int) -> Path:
    """"""
    "Создаём директорию для слов-вырезок"
    words_dir = S3_OCR_DIR / str(img_id) / 'words'
    words_dir.mkdir(parents=True, exist_ok=True)
    
    "Слово-вырезку"
    word_path = words_dir / f'word_{word_idx:04d}.png'
    word_img.save(word_path)
    return word_path


def save_result_text(text: str, img_id: int) -> Path:
    """Сохранение предсказания-текста в .txt в соответствующую директорию с изображением"""
    img_dir = S3_OCR_DIR / str(img_id)
    txt_path = img_dir / 'result.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    log_event(f"Сохранили предсказание-текст \033[32m{img_id}\033[0m to .txt")
    return txt_path


def create_archive(img_id: int) -> Path:
    """
    Создаёт tar.gz архив из директории изображения.
    
    Структура архива:
        img_id/
            original.png
            result.txt
            words/
                word_0001.png
                word_0002.png
                ...
    
    Args:
        img_id: идентификатор изображения
        
    Returns:
        Path к созданному архиву
    """
    img_dir = S3_OCR_DIR / str(img_id)
    if not img_dir.exists():
        raise FileNotFoundError(f"Directory {img_dir} not found")
    
    "Создание архива"
    archive_path = S3_OCR_DIR / f'{img_id}.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as tar:
        # Добавляем всю директорию в архив
        tar.add(img_dir, arcname=str(img_id))
    log_event(f"Создан архив для img \033[31m{img_id}\033[0m", level='WARNING')
    
    "Удаление директории после архивации"
    import shutil
    shutil.rmtree(img_dir)
    
    return archive_path


def move_archive_as_succeeded(archive_path: Path):
    """
    Перемещаем архив из рабочей директории как "успешно перенесённый"
    """
    if archive_path.exists():
        shutil.move(archive_path, S3_TEMP_DIR)
        log_event(f"Архив перемещён: \033[34m{archive_path}\033[0m", level='WARNING')


async def process_batch_images(
        files: list[UploadFile],
        img_ids: list[int]

) -> list[tuple[int, Path, Image.Image]]:
    """
    Обрабатывает батч изображений: сохраняет на диск и загружает в память.
    
    Args:
        files: список загруженных файлов
        img_ids: список идентификаторов
        
    Returns:
        Список кортежей (img_id, img_path, img_pil)
    """
    if len(files) != len(img_ids):
        raise ValueError(f"Number of files ({len(files)}) != number of img_ids ({len(img_ids)})")
    
    results = []
    
    for file, img_id in zip(files, img_ids):
        "Сохранение на диск"
        img_path = await save_uploaded_image(file, img_id)
        
        "Загружаем в память для инференса"
        img_pil = load_image_from_path(img_path)
        
        results.append((img_id, img_path, img_pil))
    
    log_event(f"Подготовили batch из \033[33{len(results)}\033[0m images", level='INFO')
    return results
