import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from ml.dataclass_detector import OCRDetectorDataset


yolov8n = YOLO('yolov8n.yaml')
m = yolov8n.model

# 1. Меняем входной слой
old_conv = m.model[0].conv
m.model[0].conv = nn.Conv2d(
    1, old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# 2. Настраиваем классы
nc_names = OCRDetectorDataset.classes
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


model_detector = m

model_detector_code = '''
import torch
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import Detect

from ml.dataclass_detector import OCRDetectorDataset


yolov8n = YOLO('yolov8n.yaml')
m = yolov8n.model

# 1. Меняем входной слой
old_conv = m.model[0].conv
m.model[0].conv = nn.Conv2d(
    1, old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None
)

# 2. Настраиваем классы
nc_names = OCRDetectorDataset.classes
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


model_detector = m
'''