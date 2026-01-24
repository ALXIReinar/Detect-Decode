import torch
from torch import nn

from ml.loss_function.utils import bbox_ciou, dfl_decode


class OCRYOLOv8Loss(nn.Module):
    def __init__(self, reg_max=16, box_weight=7.5, obj_weight=1.0, topk=10):
        super().__init__()
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.topk = topk
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """
        preds:
            boxes: [B, reg_max, N_anchors]  <- распределение DFL
            scores: [B, 1, N_anchors] или [B, N_anchors]
        targets:
            list of [Mi, 4] (xyxy)
        """
        B = preds["boxes"].shape[0]

        # --- DFL decode: [B, reg_max, N] -> [B, N, 4] ---
        pred_boxes = dfl_decode(preds["boxes"], self.reg_max)  # [B, N_anchors, 4]

        # Если scores [B,1,N] -> squeeze до [B,N]
        scores = preds["scores"]
        if scores.ndim == 3 and scores.shape[1] == 1:
            scores = scores.squeeze(1)  # [B, N_anchors]

        total_box_loss = 0.0
        total_obj_loss = 0.0

        for b in range(B):
            boxes = pred_boxes[b]  # [N_anchors,4]
            score = scores[b]      # [N_anchors]
            gt = targets[b]        # [Mi,4]

            if gt.numel() == 0:
                total_obj_loss += self.bce(score, torch.zeros_like(score))
                continue

            # --- CIoU матрица ---
            ciou = bbox_ciou(boxes, gt)  # [N_anchors, Mi]

            pos_mask = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
            iou_target = torch.zeros(boxes.shape[0], device=boxes.device)

            # --- mini Task-Aligned Assignment ---
            for j in range(gt.size(0)):
                iou_j = ciou[:, j]
                k = min(self.topk, iou_j.numel())
                topk_iou, topk_idx = torch.topk(iou_j, k)
                valid = topk_iou > 0
                idx = topk_idx[valid]

                pos_mask[idx] = True
                iou_target[idx] = torch.maximum(iou_target[idx], topk_iou[valid])

            # --- CIoU loss по positive ---
            if pos_mask.any():
                total_box_loss += (1.0 - iou_target[pos_mask]).mean()

            # --- Soft objectness ---
            total_obj_loss += self.bce(score, iou_target)

        total_loss = (
            self.box_weight * total_box_loss +
            self.obj_weight * total_obj_loss
        ) / B

        return total_loss, {
            "box": total_box_loss.detach(),
            "obj": total_obj_loss.detach(),
        }
