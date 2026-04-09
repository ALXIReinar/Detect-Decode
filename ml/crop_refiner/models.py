from typing import Literal

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
from ultralytics.utils.metrics import box_iou
from ultralytics.utils.ops import xywh2xyxy


class Extent2CoreRefiner(nn.Module):
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


class IoULoss(nn.Module):
    """IoU Loss для регрессии bounding boxes."""

    def __init__(self, format_bbox: Literal['XYXY', 'XYWH']):
        super().__init__()
        self.format_bbox = format_bbox

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (B, 4) - предсказанные боксы [cx, cy, w, h]
            target_boxes: (B, 4) - ground truth боксы [cx, cy, w, h]

        Returns:
            loss: скаляр
        """
        if self.format_bbox == 'XYWH':
            pred_boxes, target_boxes = xywh2xyxy(pred_boxes), xywh2xyxy(target_boxes)

        iou = box_iou(pred_boxes, target_boxes)
        loss = 1 - iou.mean()
        return loss