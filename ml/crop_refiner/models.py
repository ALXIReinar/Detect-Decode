import math
from typing import Literal

import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights, resnet18, ResNet18_Weights
from torch import nn
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy



class Extent2CoreMobileNetRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(576, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        return self.regressor(x)


class Extent2CoreResnetRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)



class IoULoss(nn.Module):
    """IoU Loss для регрессии bounding boxes."""

    def __init__(self, format_bbox: Literal['XYXY', 'XYWH']):
        super().__init__()
        self.format_bbox = format_bbox

    def forward(self, pred_boxes, target_boxes):
        if self.format_bbox == 'XYWH':
            pred_boxes, target_boxes = xywh2xyxy(pred_boxes), xywh2xyxy(target_boxes)

        iou = box_iou(pred_boxes, target_boxes)
        loss = 1 - iou.mean()
        return loss


class CIoULoss(nn.Module):
    def __init__(self, format_bbox: Literal['XYXY', 'XYWH'], eps: float = 1e-7):
        super().__init__()
        self.format_bbox = format_bbox
        self.eps = eps  # Защита от деления на ноль

    def forward(self, pred_boxes, target_boxes):
        # 1. Приводим всё к форматам XYXY и извлекаем w, h
        if self.format_bbox == 'XYWH':
            # Предполагается, что на вход подаются: cx, cy, w, h
            pred_xyxy = xywh2xyxy(pred_boxes)
            target_xyxy = xywh2xyxy(target_boxes)

            w_pred, h_pred = pred_boxes[:, 2], pred_boxes[:, 3]
            w_target, h_target = target_boxes[:, 2], target_boxes[:, 3]
        else:
            # На вход подаются: xmin, ymin, xmax, ymax
            pred_xyxy = pred_boxes
            target_xyxy = target_boxes

            w_pred = pred_xyxy[:, 2] - pred_xyxy[:, 0]
            h_pred = pred_xyxy[:, 3] - pred_xyxy[:, 1]
            w_target = target_xyxy[:, 2] - target_xyxy[:, 0]
            h_target = target_xyxy[:, 3] - target_xyxy[:, 1]

        # 2. Вычисляем поэлементный IoU
        inter_xmin = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
        inter_ymin = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
        inter_xmax = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
        inter_ymax = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])

        inter_w = torch.clamp(inter_xmax - inter_xmin, min=0)
        inter_h = torch.clamp(inter_ymax - inter_ymin, min=0)
        inter_area = inter_w * inter_h

        area_pred = w_pred * h_pred
        area_target = w_target * h_target
        union_area = area_pred + area_target - inter_area + self.eps

        iou = inter_area / union_area

        # 3. Вычисляем Center Distance Penalty (расстояние между центрами)
        cx_pred = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        cy_pred = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        cx_target = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
        cy_target = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2

        # Квадрат расстояния между центрами
        rho2 = (cx_pred - cx_target) ** 2 + (cy_pred - cy_target) ** 2

        # Диагональ наименьшего объемлющего прямоугольника (Convex Box)
        convex_xmin = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        convex_ymin = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        convex_xmax = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        convex_ymax = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])

        c2 = (convex_xmax - convex_xmin) ** 2 + (convex_ymax - convex_ymin) ** 2 + self.eps

        # 4. Вычисляем Aspect Ratio Penalty (v)
        # Арктангенсы пропорций
        atan_target = torch.atan(w_target / (h_target + self.eps))
        atan_pred = torch.atan(w_pred / (h_pred + self.eps))

        v = (4 / (math.pi ** 2)) * torch.pow((atan_target - atan_pred), 2)

        # 5. Вычисляем Trade-off параметр (alpha)
        # alpha не участвует в обратном распространении ошибки (по оригинальной статье)
        with torch.no_grad():
            alpha = v / (1 - iou + v + self.eps)

        # 6. Итоговый CIoU и Loss
        ciou = iou - (rho2 / c2) - (alpha * v)
        loss = 1 - ciou

        return loss.mean()