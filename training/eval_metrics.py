import numpy as np
from sklearn.metrics import roc_auc_score

def iou(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    union = (pred | gt).sum() + eps
    return float(inter) / float(union)

def dice(pred, gt, eps=1e-6):
    inter = (pred & gt).sum()
    return float((2*inter + eps) / (pred.sum() + gt.sum() + eps))

def roc_auc(image_flags, labels):
    # image_flags: list of scores (e.g., max_prob); labels: 0/1
    return roc_auc_score(labels, image_flags)