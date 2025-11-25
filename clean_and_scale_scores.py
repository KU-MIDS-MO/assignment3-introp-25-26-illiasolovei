# 1) clean_and_scale_scores(scores, min_score, max_score)
#    We have exam scores stored in a NumPy array (1D or 2D).
#    - First, replace all values smaler than min_score by min_score,
#      and all values larger than max_score by max_score.

#    - Then linearly scale all values to the range [0, 1] us ing:
#        scaled = (value - min_score) / (max_score - min_score)
#    Return a new NumPy array of floats with the same shape as scores

import numpy as np


def toFloat(n):
    return n + 0.0


def clean_and_scale_scores(scores, min_score, max_score):
    shp = scores.shape
    scores_flat = scores.reshape(scores.size)
    new = np.zeros(scores_flat.shape, dtype=float)
    den = toFloat(max_score - min_score)

    for i in range(scores_flat.size):
        score = scores_flat[i]
        if score < min_score:
            score = min_score
        elif score > max_score:
            score = max_score

        diff = score - min_score
        new[i] = diff / den

    return new.reshape(shp)
