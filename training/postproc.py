import numpy as np
import cv2

def binarize(probs, thr=0.5, min_area_px=50):
    mask = (probs >= thr).astype(np.uint8)
    # remove tiny blobs
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
            clean[cc == i] = 1
    return clean

def scores_from_mask(prob_map, bin_mask):
    edited_area_pct = 100.0 * bin_mask.sum() / (bin_mask.shape[0]*bin_mask.shape[1])
    max_prob = float(prob_map.max())
    # focus on largest blob
    num, cc, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), 8)
    mean_prob_largest = 0.0
    if num > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mean_prob_largest = float(prob_map[cc==largest_idx].mean())
    return dict(
        edited_area_pct=edited_area_pct,
        max_prob=max_prob,
        mean_prob_largest=mean_prob_largest
    )

def flag_decision(scores, area_min_pct=0.2, maxprob_min=0.8, meanprob_min=0.6):
    # Example heuristic — tune on validation set
    cond1 = scores["max_prob"] >= maxprob_min and scores["edited_area_pct"] >= area_min_pct
    cond2 = scores["edited_area_pct"] >= 1.0 and scores["mean_prob_largest"] >= meanprob_min
    return bool(cond1 or cond2)