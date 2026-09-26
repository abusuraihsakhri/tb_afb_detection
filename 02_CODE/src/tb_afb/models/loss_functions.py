import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for logits.

    alpha weights the positive class while 1 - alpha weights the negative
    class, following the original focal-loss formulation.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum.")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=inputs.dtype)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probability_correct = torch.exp(-bce)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * (1.0 - probability_correct).pow(self.gamma) * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CIOULoss(nn.Module):
    """Complete-IoU loss for boxes in (cx, cy, width, height) format."""

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be one of: none, mean, sum.")
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        if pred_boxes.shape != target_boxes.shape or pred_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes and target_boxes must have matching (..., 4) shapes.")

        pred_xy = pred_boxes[..., :2]
        target_xy = target_boxes[..., :2]
        pred_wh = pred_boxes[..., 2:].clamp_min(self.eps)
        target_wh = target_boxes[..., 2:].clamp_min(self.eps)

        pred_min = pred_xy - pred_wh / 2.0
        pred_max = pred_xy + pred_wh / 2.0
        target_min = target_xy - target_wh / 2.0
        target_max = target_xy + target_wh / 2.0

        inter_min = torch.maximum(pred_min, target_min)
        inter_max = torch.minimum(pred_max, target_max)
        inter_wh = (inter_max - inter_min).clamp_min(0.0)
        intersection = inter_wh[..., 0] * inter_wh[..., 1]

        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        target_area = target_wh[..., 0] * target_wh[..., 1]
        union = pred_area + target_area - intersection
        iou = intersection / (union + self.eps)

        center_distance_sq = ((pred_xy - target_xy) ** 2).sum(dim=-1)
        enclosing_min = torch.minimum(pred_min, target_min)
        enclosing_max = torch.maximum(pred_max, target_max)
        enclosing_diagonal_sq = ((enclosing_max - enclosing_min) ** 2).sum(dim=-1) + self.eps

        pred_ratio = torch.atan(pred_wh[..., 0] / pred_wh[..., 1])
        target_ratio = torch.atan(target_wh[..., 0] / target_wh[..., 1])
        v = (4.0 / math.pi**2) * (target_ratio - pred_ratio).pow(2)
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + self.eps)

        ciou = iou - center_distance_sq / enclosing_diagonal_sq - alpha * v
        loss = 1.0 - ciou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
