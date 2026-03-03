import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from ml.detector.dataset_class.dataclass_detector import OCRDetectorDataset

"Более грязный способ использования"
# yolov8n = YOLO('yolov8n.yaml')
# m = yolov8n.model
#
# # 1. Меняем входной слой
# old_conv = m.model[0].conv
# m.model[0].conv = torch.nn.Conv2d(
#     1, old_conv.out_channels,
#     kernel_size=old_conv.kernel_size,
#     stride=old_conv.stride,
#     padding=old_conv.padding,
#     bias=old_conv.bias is not None
# )
#
# "Настраиваем классы"
# nc_names = OCRDetectorDataset('', auto_load=False).classes
# new_nc = len(nc_names)
# m.nc = new_nc
# m.classes = {idx: cls_n for idx, cls_n in enumerate(nc_names)}
#
# "Пересобираем Detect"
# old_detect = m.model[-1]
#
# "В YOLOv8n входы в голову: [64, 128, 256]"
# input_channels = tuple(neck_prt[0].conv.in_channels for neck_prt in old_detect.cv2)
#
# new_detect = Detect(nc=new_nc, ch=input_channels)
#
# "Добираем необходимые атрибуты"
# new_detect.stride = old_detect.stride
# new_detect.inplace = old_detect.inplace
# new_detect.f = old_detect.f
# new_detect.i = old_detect.i
# new_detect.type = old_detect.type
#
#
# m.model[-1] = new_detect
#
# "Настраиваем страйд, чтобы модель не смывала bboxes"
# new_detect.stride = torch.tensor([8., 16., 32.])
#
#
# model_detector = m

class WordDetector(nn.Module):
    def __init__(self):
        super().__init__()
        yolov8n = YOLO('yolov8n.yaml')
        m = yolov8n.model

        # 1. Меняем входной слой
        old_conv = m.model[0].conv
        m.model[0].conv = torch.nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        "Настраиваем классы"
        nc_names = OCRDetectorDataset('', auto_load=False).classes
        new_nc = len(nc_names)
        m.nc = new_nc
        m.classes = {idx: cls_n for idx, cls_n in enumerate(nc_names)}

        "Пересобираем Detect"
        old_detect = m.model[-1]

        "В YOLOv8n входы в голову: [64, 128, 256]"
        input_channels = tuple(neck_prt[0].conv.in_channels for neck_prt in old_detect.cv2)

        new_detect = Detect(nc=new_nc, ch=input_channels)

        "Добираем необходимые атрибуты"
        new_detect.stride = old_detect.stride
        new_detect.inplace = old_detect.inplace
        new_detect.f = old_detect.f
        new_detect.i = old_detect.i
        new_detect.type = old_detect.type

        m.model[-1] = new_detect

        "Настраиваем страйд, чтобы модель не смывала bboxes"
        new_detect.stride = torch.tensor([8., 16., 32.])
        
        # Регистрируем все слои YOLO модели напрямую в WordDetector
        self.model = m.model  # m.model == nn.Sequential со всеми слоями
        
        # Копируем ВСЕ важные атрибуты из YOLO DetectionModel
        self.stride = new_detect.stride
        self.names: dict[int, str] = m.names if hasattr(m, 'names') else {idx: cls_n for idx, cls_n in enumerate(nc_names)}
        self.nc: int = new_nc
        self.yaml = m.yaml if hasattr(m, 'yaml') else None
        self.args = m.args if hasattr(m, 'args') else None
        
        # КРИТИЧНО: Копируем save - список индексов слоёв для сохранения выходов
        # Это нужно для Concat модулей, которые используют выходы предыдущих слоёв
        self.save = m.save if hasattr(m, 'save') else []
        
        # Копируем inplace флаг
        self.inplace = m.inplace if hasattr(m, 'inplace') else True

    def forward(self, x):
        """
        Forward pass через YOLO модель.
        Использует логику из DetectionModel для правильной обработки Concat слоёв.
        """
        y = []  # Выходы слоёв для сохранения
        
        for i, module in enumerate(self.model):
            # Если модуль использует выходы предыдущих слоёв (Concat, Detect)
            if hasattr(module, 'f') and module.f != -1:
                # module.f - индексы слоёв, чьи выходы нужны
                if isinstance(module.f, int):
                    x = y[module.f]  # Один вход
                else:
                    # Несколько входов (для Concat)
                    x = [x if j == -1 else y[j] for j in module.f]
            
            x = module(x)  # Forward через модуль
            
            # Сохраняем выход если нужно
            if i in self.save:
                y.append(x)
            else:
                y.append(None)
        
        return x



model_detector_code = '''
"Более грязный способ использования"
# yolov8n = YOLO('yolov8n.yaml')
# m = yolov8n.model
# 
# # 1. Меняем входной слой
# old_conv = m.model[0].conv
# m.model[0].conv = torch.nn.Conv2d(
#     1, old_conv.out_channels,
#     kernel_size=old_conv.kernel_size,
#     stride=old_conv.stride,
#     padding=old_conv.padding,
#     bias=old_conv.bias is not None
# )
# 
# "Настраиваем классы"
# nc_names = OCRDetectorDataset('', auto_load=False).classes
# new_nc = len(nc_names)
# m.nc = new_nc
# m.classes = {idx: cls_n for idx, cls_n in enumerate(nc_names)}
# 
# "Пересобираем Detect"
# old_detect = m.model[-1]
# 
# "В YOLOv8n входы в голову: [64, 128, 256]"
# input_channels = tuple(neck_prt[0].conv.in_channels for neck_prt in old_detect.cv2)
# 
# new_detect = Detect(nc=new_nc, ch=input_channels)
# 
# "Добираем необходимые атрибуты"
# new_detect.stride = old_detect.stride
# new_detect.inplace = old_detect.inplace
# new_detect.f = old_detect.f
# new_detect.i = old_detect.i
# new_detect.type = old_detect.type
# 
# 
# m.model[-1] = new_detect
# 
# "Настраиваем страйд, чтобы модель не смывала bboxes"
# new_detect.stride = torch.tensor([8., 16., 32.])
# 
# 
# model_detector = m

class WordDetector(nn.Module):
    def __init__(self):
        super().__init__()
        yolov8n = YOLO('yolov8n.yaml')
        m = yolov8n.model

        # 1. Меняем входной слой
        old_conv = m.model[0].conv
        m.model[0].conv = torch.nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        "Настраиваем классы"
        nc_names = OCRDetectorDataset('', auto_load=False).classes
        new_nc = len(nc_names)
        m.nc = new_nc
        m.classes = {idx: cls_n for idx, cls_n in enumerate(nc_names)}

        "Пересобираем Detect"
        old_detect = m.model[-1]

        "В YOLOv8n входы в голову: [64, 128, 256]"
        input_channels = tuple(neck_prt[0].conv.in_channels for neck_prt in old_detect.cv2)

        new_detect = Detect(nc=new_nc, ch=input_channels)

        "Добираем необходимые атрибуты"
        new_detect.stride = old_detect.stride
        new_detect.inplace = old_detect.inplace
        new_detect.f = old_detect.f
        new_detect.i = old_detect.i
        new_detect.type = old_detect.type

        m.model[-1] = new_detect

        "Настраиваем страйд, чтобы модель не смывала bboxes"
        new_detect.stride = torch.tensor([8., 16., 32.])
        
        # Регистрируем все слои YOLO модели напрямую в WordDetector
        self.model = m.model  # m.model == nn.Sequential со всеми слоями
        
        # Копируем ВСЕ важные атрибуты из YOLO DetectionModel
        self.stride = new_detect.stride
        self.names: dict[int, str] = m.names if hasattr(m, 'names') else {idx: cls_n for idx, cls_n in enumerate(nc_names)}
        self.nc: int = new_nc
        self.yaml = m.yaml if hasattr(m, 'yaml') else None
        self.args = m.args if hasattr(m, 'args') else None
        
        # КРИТИЧНО: Копируем save - список индексов слоёв для сохранения выходов
        # Это нужно для Concat модулей, которые используют выходы предыдущих слоёв
        self.save = m.save if hasattr(m, 'save') else []
        
        # Копируем inplace флаг
        self.inplace = m.inplace if hasattr(m, 'inplace') else True

    def forward(self, x):
        """
        Forward pass через YOLO модель.
        Использует логику из DetectionModel для правильной обработки Concat слоёв.
        """
        y = []  # Выходы слоёв для сохранения
        
        for i, module in enumerate(self.model):
            # Если модуль использует выходы предыдущих слоёв (Concat, Detect)
            if hasattr(module, 'f') and module.f != -1:
                # module.f - индексы слоёв, чьи выходы нужны
                if isinstance(module.f, int):
                    x = y[module.f]  # Один вход
                else:
                    # Несколько входов (для Concat)
                    x = [x if j == -1 else y[j] for j in module.f]
            
            x = module(x)  # Forward через модуль
            
            # Сохраняем выход если нужно
            if i in self.save:
                y.append(x)
            else:
                y.append(None)
        
        return x
'''