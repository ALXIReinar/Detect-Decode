import os

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2


main_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.uint8, scale=True),
        v2.Grayscale(),
        v2.Resize((648, 648)), # ResNet-like input X 2
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(std=(0.5,), mean=(0.5,)),   # для более точных значений - https://share.google/aimode/1bhH2c9qWe2aGI8ia
    ]
)


train_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.uint8, scale=True),
        v2.Grayscale(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(
            degrees=0,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2)
        ),
        v2.ColorJitter(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.2, 0.2),
        ),
        v2.Resize((648, 648)),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(std=(0.5,), mean=(0.5,)),
    ]
)


class OCRCRNNDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.classes = ... # Сделать/найти файлик, в котором будут лежать буквы русского алфавита
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.idx_to_class = {idx: class_name for idx, class_name in enumerate(self.classes)}

        self.transform = transform
        self.data = []

    def create_data(self):
        for dirpath, dirnames, filenames in os.walk(self.path):
            ...
        # можно подумать над тем, чтобы делать "self.data[idx] = sample", чтобы было быстрее, чем "self.data.append(sample)"


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, one_hot = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, one_hot
