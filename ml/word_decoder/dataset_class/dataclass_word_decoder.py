"""
Датасет класс для CRNN распознавания рукописного текста.

Структура датасета:
    dataset/iam-words/
    ├── train/
    │   ├── imgs/
    │   │   └── a01-000u_hello_0.png
    │   └── annotations/
    │       └── a01-000u_hello_0.txt  (содержит "hello")
    └── test/
        └── ...
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
import albumentations as A

from torch.utils.data import Dataset
from torchvision.transforms import v2

from ml.config import WORKDIR
from ml.logger_config import log_event
from ml.word_decoder.utils import AddGaussianNoise


@lru_cache
def read_charset_file_lru(f_path: Path):
    if f_path.exists():
        return f_path.read_text(encoding='utf-8').split('\n')
    raise FileNotFoundError(f'File {f_path} not found.')


def read_charset(charset_input: str | list[str] | Path) -> list[str]:
    """
    Читает charset из файла или возвращает переданный список.
    """
    # Если передан список - возвращаем как есть
    if isinstance(charset_input, list):
        return charset_input
    
    # Если передан путь - читаем файл с кэшированием
    return read_charset_file_lru(Path(charset_input))


class CRNNWordAugment(nn.Module):
    def __init__(self, mode: Literal['train', 'val', 'test'], img_height: int = 64):
        """
        Args:
            mode: 'train', 'val' или 'test'
            img_height: фиксированная высота изображения (НО ширина переменная)
        """
        super().__init__()
        self.mode = mode
        self.img_height = img_height

        "Аугментации для обучения"
        if mode == 'train':
            # Albumentations работают с numpy (H, W, C)
            self.alb_aug = A.Compose([
                # Имитация низкого разрешения (сжатие + upscale с артефактами)
                A.Downscale(scale_range=(0.25, 0.5), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.06, scale_limit=0.1, rotate_limit=5, p=0.5, border_mode=0, fill=255),
            ])

            self.torch_aug = v2.Compose([
                v2.ColorJitter(brightness=0.3, contrast=0.3),
                v2.ToDtype(dtype=torch.float32, scale=True),
                AddGaussianNoise(mean=0.0, std=0.02),
            ])


    def forward(self, img: Image.Image) -> torch.Tensor:
        """"""
        "Сначала Resize + сохранение пропорций"
        w, h = img.size
        new_w = max(16, min(int(self.img_height * (w / h)), 512))
        img = img.resize((new_w, self.img_height), Image.Resampling.BILINEAR)


        "Применяем CLAHE для улучшения контраста (всегда, не только при train)"
        # Ленивая инициализация, чтобы не пересоздавать объект каждый forward. Необходимо для share между воркерами-процессами
        if not hasattr(self, 'clahe'):
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_np = np.array(img)
        img_np = self.clahe.apply(img_np)
        img = Image.fromarray(img_np)

        "Аугментации Albumentations при train"
        if self.mode == 'train':
            img_np = np.array(img)
            augmented = self.alb_aug(image=img_np)['image']
            if len(augmented.shape) == 3 and augmented.shape[2] == 1:
                augmented = augmented.squeeze(2)

            img = Image.fromarray(augmented)

        "np_img2tensor"
        img_tensor = v2.ToImage()(img)
        img_tensor = v2.ToDtype(dtype=torch.uint8)(img_tensor)

        "Прочие аугментации"
        if self.mode == 'train':
            img_tensor = self.torch_aug(img_tensor)
        else:
            img_tensor = v2.ToDtype(dtype=torch.float32, scale=True)(img_tensor)

        "Дублируем каналы для ResNet"
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)

        return img_tensor


class CRNNWordDataset(Dataset):
    """
    Датасет для CRNN распознавания рукописных слов.
    """
    
    def __init__(
        self,
        path: str | Path,
        charset_path: str | Path | list[str],
        img_height: int = 64,
        transform: Literal['train', 'val', 'test'] | None = None,
        auto_load: bool = True
    ):
        """
        Args:
            path: путь к датасету (например, 'dataset/iam-words/train')
            charset_path: путь к файлу набора символов
            img_height: фиксированная высота изображения
            transform: режим трансформаций ('train', 'val', 'test' или None)
            auto_load: автоматически загружать данные при инициализации (по умолчанию True)
        """
        self.path = Path(path)
        self.charset: list[str] = read_charset(charset_path)
        self.img_height = img_height
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        
        self.transform = CRNNWordAugment(transform, img_height) if transform else None
        self.imagenet_normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet Normalize
        self.data = []
        
        "Автоматическая загрузка данных"
        if auto_load:
            self.load_data()

    
    def load_data(self):
        """
        Загружает данные из директории.
        
        Returns:
            self: возвращает сам объект для chain calling
        """
        data = []
        
        imgs_dir = self.path / 'imgs'
        anns_dir = self.path / 'annotations'
        
        if not imgs_dir.exists() or not anns_dir.exists():
            raise ValueError(f"Директории не найдены: {imgs_dir} или {anns_dir}")
        
        "Проход по изображениям"
        for img_path in sorted(imgs_dir.glob('*.png')):
            ann_path = anns_dir / f"{img_path.stem}.txt"
            
            if not ann_path.exists():
                continue
            
            # Читаем текст
            with open(ann_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if not text:
                continue
            
            # Проверяем, что все символы из алфавита
            if not all(char in self.char_to_idx for char in text):
                continue
            
            data.append({
                'img_path': img_path,
                'text': text
            })
        
        if not data:
            raise ValueError(f"Не найдено валидных семплов в {self.path}")

        log_event(f"Загружено \033[31m{len(data)}\033[0m семплов из \033[34m{self.path}\033[0m")
        self.data = data
        
        return self
    
    def text_to_indices(self, text: str) -> list[int]:
        """
        Конвертирует текст в список индексов.
        
        Args:
            text: текст (например, "hello")
            
        Returns:
            список индексов (например, [8, 5, 12, 12, 15])
        """
        return [self.char_to_idx[char] for char in text]
    
    def indices_to_text(self, indices: list[int]) -> str:
        """
        Конвертирует список индексов в текст.
        
        Args:
            indices: список индексов
            
        Returns:
            текст
        """
        return ''.join(self.idx_to_char[idx] for idx in indices if idx != 0)  # пропускаем <blank>


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Возвращает один семпл.
        
        Args:
            idx: индекс семпла
            
        Returns:
            (img, text_indices, text_length)
            - img: тензор изображения [3, H, W] - 3 канала для ResNet
            - text_indices: тензор индексов символов [text_length]
            - text_length: длина текста (int)
        """
        sample = self.data[idx]
        
        # Загружаем изображение (grayscale для OCR)
        img = Image.open(sample['img_path']).convert('L')
        
        # Применяем трансформации
        if self.transform:
            img = self.transform(img)
        else:
            # Базовая обработка без аугментаций
            img = v2.ToImage()(img)
            img = v2.ToDtype(dtype=torch.float32, scale=True)(img)
            
            # Resize с сохранением aspect ratio
            _, h, w = img.shape
            aspect_ratio = w / h
            new_width = int(self.img_height * aspect_ratio)
            new_width = max(16, min(new_width, 512))
            img = v2.Resize((self.img_height, new_width), antialias=True)(img)
            
            # Дублируем grayscale канал в 3 канала для ResNet
            # [1, H, W] -> [3, H, W]
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
        
        # text2indexes
        text = sample['text']
        text_indices = self.text_to_indices(text)
        text_length = len(text_indices)
        
        text_indices = torch.tensor(text_indices, dtype=torch.long)
        return img, text_indices, text_length, img.shape[2]
    
    @staticmethod
    def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate функция для DataLoader.
        Паддит изображения до одной ширины и конкатенирует тексты.

        Args:
            batch: список из (img, text_indices, text_length)

        Returns:
            (images, targets, input_lengths, target_lengths)
            - images: [batch, 3, H, max_width] - паддированные изображения (3 канала)
            - targets: [sum(target_lengths)] - конкатенированные индексы символов
            - input_lengths: [batch] - длины последовательностей после модели
            - target_lengths: [batch] - длины целевых текстов
        """
        images = []
        all_text_indices = []
        target_lengths = []
        orig_lengths = []
        
        # Находим максимальную ширину в батче
        max_width = max(img.shape[2] for img, _, _, _ in batch)
        
        for img, text_indices, text_length, orig_length in batch:
            # Паддим изображение справа до max_width
            c, h, w = img.shape  # [3, H, W]
            if w < max_width:
                # Создаём чёрный паддинг (значение 0.0) для всех каналов
                padded_img = torch.zeros(c, h, max_width, dtype=img.dtype)
                padded_img[:, :, :w] = img
                img = padded_img
            
            images.append(img)
            all_text_indices.append(text_indices)
            target_lengths.append(text_length)
            orig_lengths.append(orig_length)
        
        # Стекаем изображения + Normalize
        images = torch.stack(images, dim=0)  # [batch, C, H, max_width]
        norm = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # ImageNet Normalize
        images = norm(images)

        # Конкатенируем все text_indices
        targets = torch.cat(all_text_indices, dim=0)  # [sum(target_lengths)]
        
        # Длины целевых текстов
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)  # [batch]
        
        # Так как паддим изображение, нужно сохранить оригинальную ширину слова // 4, так как даунсемпл от ResNet backbone
        input_lengths = torch.tensor(orig_lengths, dtype=torch.long) // 4
        
        return images, targets, input_lengths, target_lengths


# Пример использования
if __name__ == "__main__":
    # Создаём датасет
    train_dataset = CRNNWordDataset(
        path=WORKDIR / 'dataset/iam-words/test',
        charset_path=WORKDIR / 'dataset/iam-words/charset.txt',
        img_height=64,
        transform='train'
    )
    train_dataset.load_data()

    print(f"\nДатасет: {len(train_dataset)} семплов")

    # Тестируем один семпл
    img, text_indices, text_length, orig_lengths = train_dataset[0]
    print(f"\nПример семпла:")
    print(f"  Изображение: {img.shape} (должно быть [3, 64, W])")
    print(f"  Каналы: {img.shape[0]} (должно быть 3 для ResNet)")
    print(f"  Текст (индексы): {text_indices}")
    print(f"  Текст (строка): '{train_dataset.indices_to_text(text_indices.tolist())}'")
    print(f"  Длина текста: {text_length}")

    # Тестируем DataLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=CRNNWordDataset.collate_fn,
        num_workers=0
    )

    print(f"\nТест DataLoader:")
    images, targets, input_lengths, target_lengths = next(iter(train_loader))
    print(f"  Images: {images.shape} (должно быть [8, 3, 64, W])")
    print(f"  Targets: {targets.shape}")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Target lengths: {target_lengths}")

    # Тестируем один семпл
    img, text_indices, text_length, input_lengths = train_dataset[0]
    print(f"\nПример семпла:")
    print(f"  Изображение: {img.shape}")
    print(f"  Текст (индексы): {text_indices}")
    print(f"  Текст (строка): '{train_dataset.indices_to_text(text_indices.tolist())}'")
    print(f"  Длина текста: {text_length}")
