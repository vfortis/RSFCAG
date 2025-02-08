from scipy.spatial.distance import cdist
import numpy as np


def MystepRFCM(centers, X, expo):
    '''
    基于阴影集的粗糙模糊聚类算法的隶属度矩阵的计算
    :param centers:
    :param X:
    :param expo:
    :return: 隶属度矩阵
    '''
    distance = cdist(X, centers, metric='euclidean')

    U_new = np.where(np.sum(distance == 0, axis=1, keepdims=True) >= 1, np.where(distance == 0, 1, 0), 1 / (distance ** (1 / (expo - 1)) * np.sum(1 / distance ** (1 / (expo - 1)), axis=1, keepdims=True)))

    return U_new