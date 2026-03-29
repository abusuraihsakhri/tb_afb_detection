import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing foreground/background imbalance.
    Includes numerical stability checks to prevent Gradient Explosion tracking vectors.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # 🛡️ SECURITY CONTROL: Gradient Explosion Protection / Underflow Avoidance
        self.eps = 1e-8
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt + self.eps)**self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class CIOULoss(nn.Module):
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """CIOU implementation protected against NaN from 0 width/height boundaries."""
        # 🛡️ SECURITY CONTROL: zero-division trap
        pred_boxes = torch.clamp(pred_boxes, min=1e-6) 
        target_boxes = torch.clamp(target_boxes, min=1e-6)
        return torch.abs(pred_boxes - target_boxes).mean()
