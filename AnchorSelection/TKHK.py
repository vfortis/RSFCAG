from .SRFCM import SRFCM
from .HKM3 import HKM3
import numpy as np


def TKHK(X, numanchor, iters=100, error=1e-4):
    labels = SRFCM(X, iters, error)
    id1 = np.where((labels[0] == 0) | (labels[0] == 1))[0]
    id2 = np.where((labels[1] == 0) | (labels[1] == 1))[0]
    id3 = np.where((labels[2] == 0) | (labels[2] == 1))[0]
    centers1 = HKM3(X, id1, numanchor-1)
    centers2 = HKM3(X, id2, numanchor-1)
    centers3 = HKM3(X, id3, numanchor-1)
    centers = np.vstack((centers1, centers2, centers3))
    has_nan_row = np.isnan(centers).any(axis=1)
    new_centers = centers[~has_nan_row, :]
    return new_centers
