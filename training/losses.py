import torch
import torch.nn.functional as F

def soft_dice_loss(logits, targets, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2,3))
    den = (probs + targets).sum(dim=(1,2,3)) + eps
    loss = 1.0 - (num / den)
    return loss.mean()

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets)
        return self.bce_w * bce + self.dice_w * dice