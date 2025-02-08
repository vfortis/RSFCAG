import numpy as np


def EuDist2(fea_a, fea_b, bSqrt=True):
    aa = np.sum(fea_a * fea_a, axis=1)
    bb = np.sum(fea_b * fea_b, axis=1)
    ab = np.dot(fea_a, fea_b.T)

    D = aa[:, np.newaxis] + bb - 2 * ab

    # 处理舍入误差
    D[D < 0] = 0

    if bSqrt:
        D = np.sqrt(D)

    return D



