import numpy as np
from .BKM3 import BKM3


def HKM3(X, idx0, numanchor):
    '''

    :param X:
    :param idx0:
    :param numanchor: 三叉树的层数，锚点的个数是3^(numanchor)
    :param iters:
    :return:
    '''
    X0 = X[idx0, :]
    if numanchor == 1:
        centers, labels = BKM3(X0, 0.0)
    else:
        centers, labels = BKM3(X0, 1/3)
    if numanchor > 1:
        id1 = np.where(labels == 0)
        idx1 = idx0[id1]
        centers1 = HKM3(X, idx1, numanchor - 1)

        id2 = np.where(labels == 1)
        idx2 = idx0[id2]
        centers2 = HKM3(X, idx2, numanchor - 1)

        id3 = np.where(labels == 2)
        idx3 = idx0[id3]
        centers3 = HKM3(X, idx3, numanchor - 1)

        centers = np.vstack((centers1, centers2, centers3))
        # np.concatenate((centers1, centers2), axis=1)
    return centers

