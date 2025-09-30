import numpy as np
from training.postproc import binarize, scores_from_mask, flag_decision

def test_flag_decision_simple():
    prob = np.zeros((128,128), np.float32)
    prob[32:96, 40:90] = 0.95
    mask = binarize(prob, thr=0.5, min_area_px=10)
    scores = scores_from_mask(prob, mask)
    assert flag_decision(scores) is True