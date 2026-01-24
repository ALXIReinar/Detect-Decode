import torch


def bbox_ciou(box1, box2, eps=1e-7):
    """
    box1: [N, 4], box2: [M, 4] in xyxy
    returns CIoU [N, M]
    """
    # Intersection
    inter = (
        torch.min(box1[:, None, 2:], box2[None, :, 2:]) -
        torch.max(box1[:, None, :2], box2[None, :, :2])
    ).clamp(0)
    inter_area = inter[..., 0] * inter[..., 1]

    # Areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = area1[:, None] + area2[None, :] - inter_area + eps
    iou = inter_area / union

    # Centers
    c1 = (box1[:, None, :2] + box1[:, None, 2:]) / 2
    c2 = (box2[None, :, :2] + box2[None, :, 2:]) / 2
    rho2 = ((c1 - c2) ** 2).sum(-1)

    # Enclosing box
    enclose_min = torch.min(box1[:, None, :2], box2[None, :, :2])
    enclose_max = torch.max(box1[:, None, 2:], box2[None, :, 2:])
    c2_diag = ((enclose_max - enclose_min) ** 2).sum(-1) + eps

    # Aspect ratio term
    w1 = box1[:, 2] - box1[:, 0]
    h1 = box1[:, 3] - box1[:, 1]
    w2 = box2[:, 2] - box2[:, 0]
    h2 = box2[:, 3] - box2[:, 1]

    v = (4 / (torch.pi ** 2)) * (
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1[:, None] / (h1[:, None] + eps))
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c2_diag) - alpha * v

def dfl_decode(pred_dist, reg_max):
    """
    pred_dist: [B, 4*reg_max, N_anchors]
    reg_max: int
    returns: [B, N_anchors, 4] в xyxy
    """
    B, C, N = pred_dist.shape
    assert C % 4 == 0, f"Channels {C} not divisible by 4"
    reg_max_check = C // 4
    assert reg_max_check == reg_max, f"Expected reg_max={reg_max}, got {reg_max_check}"

    # reshape к [B, N, 4, reg_max]
    pred = pred_dist.view(B, 4, reg_max, N).permute(0, 3, 1, 2).contiguous()
    # теперь pred: [B, N, 4, reg_max]

    prob = pred.softmax(-1)
    proj = torch.arange(reg_max, device=pred.device, dtype=pred.dtype)

    return (prob * proj).sum(-1)  # [B, N, 4]
