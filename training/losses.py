import torch
import torch.nn.functional as F

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    # logits: (N,1,H,W); targets: (N,1,H,W) in {0,1}
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2,3)) + eps
    den = (probs**2 + targets**2).sum(dim=(2,3)) + eps
    dice = 1 - (num / den)
    return dice.mean()

def bce_dice_loss(logits, targets, bce_w: float = 0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dce = dice_loss(logits, targets)
    return bce_w * bce + (1 - bce_w) * dce