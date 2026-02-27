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

import torch
from PIL import Image
from torch import nn

from torch.utils.data import Dataset
from torchvision.transforms import v2

from ml.config import WORKDIR
from ml.logger_config import log_event



@lru_cache
def read_charset_file_lru(f_path: Path):
    if f_path.exists():
        return f_path.read_text(encoding='utf-8').split('\n')
    raise FileNotFoundError(f'File {f_path} not found.')


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
        
        self.base_transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
        ])
        self.augmentations = v2.ToDtype(dtype=torch.float32, scale=True)

        "Аугментации для обучения"
        if mode == 'train':
            self.augmentations = v2.Compose([
                v2.RandomRotation(degrees=5, fill=1.0),
                v2.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.85, 1.15),
                    shear=7,
                    fill=1.0
                ),
                
                # Размытие (имитация плохого качества сканирования)
                v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.5),
                
                # Изменение яркости и контраста
                v2.ColorJitter(brightness=0.4, contrast=0.4),
                
                # Случайная инверсия (белый текст на чёрном фоне)
                v2.RandomInvert(p=0.1),

                v2.ToDtype(dtype=torch.float32, scale=True),
            ])

    
    def resize_keep_aspect_ratio(self, img: torch.Tensor) -> torch.Tensor:
        """
        Изменяет размер изображения с сохранением aspect ratio.
        Высота фиксирована, ширина пропорциональна.
        
        Args:
            img: изображение [C, H, W]
            
        Returns:
            изображение [C, img_height, new_width]
        """
        _, h, w = img.shape
        
        # Вычисляем новую ширину с сохранением aspect ratio
        aspect_ratio = w / h
        new_width = int(self.img_height * aspect_ratio)
        
        # Ограничиваем минимальную и максимальную ширину в пикселях
        new_width = max(16, min(new_width, 512))
        
        img = v2.Resize((self.img_height, new_width), antialias=True)(img)
        return img


    def forward(self, img: Image.Image) -> torch.Tensor:
        """
        Применяет трансформации к изображению.
        
        Args:
            img: PIL изображение (grayscale)
            
        Returns:
            тензор [3, H, W] - 3 канала для ResNet
        """
        img = self.base_transform(img)
        img = self.resize_keep_aspect_ratio(img)   # Resize с сохранением aspect ratio
        img = self.augmentations(img)
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img


class CRNNWordDataset(Dataset):
    """
    Датасет для CRNN распознавания рукописных слов.
    """
    
    def __init__(
        self,
        path: str | Path,
        charset_path: str | Path,
        img_height: int = 64,
        transform: Literal['train', 'val', 'test'] | None = None
    ):
        """
        Args:
            path: путь к датасету (например, 'dataset/iam-words/train')
            charset_path: путь к файлу набора символов
            img_height: фиксированная высота изображения
            transform: режим трансформаций ('train', 'val', 'test' или None)
        """
        self.path = Path(path)
        self.charset: list[str] = read_charset_file_lru(Path(charset_path))
        self.img_height = img_height
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.charset)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.charset)}
        
        self.transform = CRNNWordAugment(transform, img_height) if transform else None
        
        self.data = self.load_data()
        
        log_event(f"Загружено \033[31m{len(self.data)}\033[0m семплов из \033[34m{self.path}\033[0m")
    
    def load_data(self) -> list[dict]:
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
        
        return data
    
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


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
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
        return img, text_indices, text_length
    
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
        
        # Находим максимальную ширину в батче
        max_width = max(img.shape[2] for img, _, _ in batch)
        
        for img, text_indices, text_length in batch:
            # Паддим изображение справа до max_width
            c, h, w = img.shape  # [3, H, W]
            if w < max_width:
                # Создаём белый паддинг (значение 1.0) для всех каналов
                padded_img = torch.ones(c, h, max_width, dtype=img.dtype)
                padded_img[:, :, :w] = img
                img = padded_img
            
            images.append(img)
            all_text_indices.append(text_indices)
            target_lengths.append(text_length)
        
        # Стекаем изображения
        images = torch.stack(images, dim=0)  # [batch, C, H, max_width]
        
        # Конкатенируем все text_indices
        targets = torch.cat(all_text_indices, dim=0)  # [sum(target_lengths)]
        
        # Длины целевых текстов
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)  # [batch]
        
        # Длины входных последовательностей (будут вычислены после forward pass модели)
        # Пока ставим заглушку, реальные значения будут после модели
        # Обычно это width // 8 (из-за downsampling в ResNet)
        input_lengths = torch.full((len(batch),), max_width // 8, dtype=torch.long)
        
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

    print(f"\nДатасет: {len(train_dataset)} семплов")

    # Тестируем один семпл
    img, text_indices, text_length = train_dataset[0]
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
    img, text_indices, text_length = train_dataset[0]
    print(f"\nПример семпла:")
    print(f"  Изображение: {img.shape}")
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
    print(f"  Images: {images.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  Input lengths: {input_lengths}")
    print(f"  Target lengths: {target_lengths}")
