from .ShadowCriterion import ShadowCriterion
import numpy as np

def ShadowLable(A):
    '''

    :param A: 所有样本属于某一个簇的隶属度
    :return: label中-1表示不属于该簇，0表示可能属于该簇，1表示确切属于该簇
    '''
    A_length = len(A)
    A_max = A.max()
    A_min = A.min()
    A_mid = (A_max + A_min) / 2

    lam_range = np.arange(A_min, A_mid, 0.001)

    V_range = [ShadowCriterion(lam, A) for lam in lam_range]

    lam_best = lam_range[np.argmin(V_range)]

    lable = np.zeros(A_length)

    lable[A <= lam_best] = -1
    lable[(A > lam_best) & (A >= 1 - lam_best)] = 1

    return lable, lam_best
