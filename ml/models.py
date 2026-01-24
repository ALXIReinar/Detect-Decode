from torch import nn
from ultralytics import YOLO
from ml.dataclass_detector import OCRDetectorDataset

yolov8n = YOLO('yolov8n.yaml')
model_detector = yolov8n.model
print(yolov8n.model, end='\n\n\n')

"Блоки оригинальной YoloV8 "
fst_conv = model_detector.model[0].conv # first input layer
cdh = model_detector.model[22].cv3 # cur_detection_head


"Кастомные блоки"
nc = len(OCRDetectorDataset.classes)
class_names = OCRDetectorDataset.classes

reg_max = 16
final_out_channels = 4 * reg_max + nc  # 64 + 1 = 65, DFL распределение для каждой координаты bbox в предсказании

new_fst_conv = nn.Conv2d(
    in_channels=1,
    out_channels=fst_conv.out_channels,
    kernel_size=fst_conv.kernel_size,
    stride=fst_conv.stride,
    padding=fst_conv.padding,
    bias=fst_conv.bias,
)


"Меняем количество предсказываемых классов(количество каналов) в Detection Head модели"
new_dh_layers = nn.ModuleList()
for i in range(len(cdh)):
    conv1 = cdh[i][0]
    conv2 = cdh[i][1]
    conv3 = nn.Conv2d(
        in_channels=cdh[i][2].in_channels,
        out_channels=cdh.anchors[i].shape[0] * final_out_channels,
        kernel_size=cdh[i][2].kernel_size,
        stride=cdh[i][2].stride,
        padding=cdh[i][2].padding
    )
    new_dh_layers.append(nn.Sequential(conv1, conv2, conv3))


"Меняем дефолтное на кастом (Fine Tuning)"
model_detector.model[0].conv = new_fst_conv
model_detector.model[22].cv3 = new_dh_layers

"Задаём классы для корректного лосса модели"
model_detector.nc = nc
model_detector.names = class_names

print(model_detector)

model_detector_code = '''
from torch import nn
from torch.nn import Conv2d
from ultralytics import YOLO
from ml.dataclass_detector import OCRDetectorDataset

yolov8n = YOLO('yolov8n.yaml')
model_detector = yolov8n.model

"Блоки оригинальной YoloV8 "
fst_conv = model_detector.model[0].conv # first input layer
cdh = model_detector.model[22].cv3 # cur_detection_head


"Кастомные блоки"
nc = len(OCRDetectorDataset.classes)
class_names = OCRDetectorDataset.classes
new_fst_conv = nn.Conv2d(
    in_channels=1,
    out_channels=fst_conv.out_channels,
    kernel_size=fst_conv.kernel_size,
    stride=fst_conv.stride,
    padding=fst_conv.padding,
    bias=fst_conv.bias,
)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, bn_eps, bn_momentum, affine, track_running_stats):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_momentum, affine=affine, track_running_stats=track_running_stats)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.silu(self.bn(self.conv(x)))
        return x

"Меняем количество предсказываемых классов(количество каналов) в Detection Head модели"
new_dh_layers = []
for i in range(len(cdh)):
    new_dh_layers.append(
        nn.Sequential(
            Conv(
                in_channels=cdh[i][0].conv.in_channels,
                out_channels=nc,
                kernel_size=cdh[i][0].conv.kernel_size,
                stride=cdh[i][0].conv.stride,
                padding=cdh[i][0].conv.padding,
                bias=cdh[i][0].conv.bias,
                bn_eps=cdh[i][0].bn.eps,
                bn_momentum=cdh[i][0].bn.momentum,
                affine=cdh[i][0].bn.affine,
                track_running_stats=cdh[0][0].bn.track_running_stats
            ),
            Conv(
                in_channels=nc,
                out_channels=nc,
                kernel_size=cdh[i][1].conv.kernel_size,
                stride=cdh[i][1].conv.stride,
                padding=cdh[i][1].conv.padding,
                bias=cdh[i][1].conv.bias,
                bn_eps=cdh[i][1].bn.eps,
                bn_momentum=cdh[i][1].bn.momentum,
                affine=cdh[i][1].bn.affine,
                track_running_stats=cdh[0][1].bn.track_running_stats
            ),
            Conv2d(
                in_channels=nc,
                out_channels=nc,
                kernel_size=cdh[i][2].kernel_size,
                stride=cdh[i][2].stride,
            )
        )
    )
new_detection_head = nn.ModuleList(new_dh_layers)


"Меняем дефолтное на кастом (Fine Tuning)"
model_detector.model[0].conv = new_fst_conv
model_detector.model[22].cv3 = new_detection_head

"Задаём классы для корректного лосса модели"
model_detector.nc = nc
model_detector.names = class_names
'''