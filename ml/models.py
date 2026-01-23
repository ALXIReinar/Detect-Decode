from torch import nn
from ultralytics import YOLO

from ml.dataclass_detector import OCRDetectorDataset

yolov8n = YOLO('yolov8n.yaml')
model_detector = yolov8n.model

fst_conv = model_detector.model[0].conv

new_fst_conv = nn.Conv2d(
    in_channels=1,
    out_channels=fst_conv.out_channels,
    kernel_size=fst_conv.kernel_size,
    stride=fst_conv.stride,
    padding=fst_conv.padding,
    bias=fst_conv.bias,
)
model_detector.model[0].conv = new_fst_conv

"Задаём классы для корректного лосса модели"
model_detector.nc = len(OCRDetectorDataset.classes)
model_detector.names = OCRDetectorDataset.classes


model_detector_code = '''
from torch import nn
from ultralytics import YOLO

from ml.dataclass_detector import OCRDetectorDataset

yolov8n = YOLO('yolov8n.yaml')
model_detector = yolov8n.model

fst_conv = model_detector.model[0].conv

new_fst_conv = nn.Conv2d(
    in_channels=1,
    out_channels=fst_conv.out_channels,
    kernel_size=fst_conv.kernel_size,
    stride=fst_conv.stride,
    padding=fst_conv.padding,
    bias=fst_conv.bias,
)
model_detector.model[0].conv = new_fst_conv

"Задаём классы для корректного лосса модели"
model_detector.nc = len(OCRDetectorDataset.classes)
model_detector.names = OCRDetectorDataset.classes
'''