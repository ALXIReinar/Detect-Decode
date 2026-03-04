"""
GPU-based аугментации для ускорения обучения детектора.
Применяются на батче после collate_fn, прямо перед передачей в модель.
"""

from torchvision.transforms import v2


class GPUBatchAugment:
    """
    Применяет аугментации на GPU для всего батча сразу.
    Это быстрее, чем на CPU, и освобождает CPU для загрузки следующих батчей.
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Аугментации, которые работают на GPU
        self.augmentations = v2.Compose([
            v2.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            v2.RandomAutocontrast(p=0.5),
        ])
    
    def __call__(self, images, targets=None):
        """
        Args:
            images: [B, C, H, W] tensor на GPU
            targets: [N, 6] tensor на GPU (batch_idx, cls, cx, cy, w, h)
        
        Returns:
            Аугментированные images и targets
        """
        if not self.training:
            return images, targets
        
        # Применяем аугментации к изображениям
        # Важно: RandomAffine с bbox требует особой обработки
        # Пока используем только pixel-level аугментации
        images = self.augmentations(images)
        
        return images, targets
    
    def train(self):
        self.training = True
        return self
    
    def eval(self):
        self.training = False
        return self
