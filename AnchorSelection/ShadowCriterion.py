import numpy as np

def ShadowCriterion(lam, A):
    '''
    给定阈值计算某一个簇的面积
    :param lam:阈值
    :param A:所有样本属于某一个簇的隶属度
    :return:面积
    '''
    sum1 = np.sum(np.where(A <= lam, A, 0))
    sum2 = np.sum(np.where((A > lam) & (A >= 1-lam), 1-A, 0))
    sum3 = np.sum(np.where((A > lam) & (A < 1-lam), 1, 0))

    return abs(sum1 + sum2 + sum3)