import torch

@torch.no_grad()
def batch_dice_iou(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    logits: (N,1,H,W)
    targets: (N,1,H,W) in {0,1}
    returns: (dice_mean, iou_mean) as Python floats
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds*targets).sum(dim=(1,2,3))
    denom = (preds + targets).sum(dim=(1,2,3))

    dice = (2*inter + 1e-6) / (denom + 1e-6)
    iou  = (inter + 1e-6) / (union + 1e-6)

    return dice.mean().item(), iou.mean().item()