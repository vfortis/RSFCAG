from .ShadowLable import ShadowLable
from .MystepRFCM import MystepRFCM
import numpy as np

def SRFCM(X, iters, error):
    '''
    生成锚点的第一层采用基于阴影集的粗糙模糊聚类算法,将样本分为三个簇,
    问题是每个簇的样本个数不一样,与基于平衡k均值的分层k均值不太一样
    :param X:
    :param iters:
    :param error:
    :return: 三个簇中样本的标签情况
    '''
    wl = 0.95  # 核心区域的权重
    expo = 2 # 模糊系数
    n, m = X.shape
    cluster_num = 3
    initU = np.random.rand(n, cluster_num)  # 初始化隶属度矩阵
    Y = np.sum(initU, axis=1)  # 归一化
    Y = np.reshape(Y, (n, 1))
    initU = initU / Y
    U = initU
    centers = np.zeros((cluster_num, m))
    for k in range(cluster_num):
        [label, ra] = ShadowLable(U[:, k])

        temp1 = U[:, k][label == 1] ** expo
        sum1 = np.sum(temp1[:, np.newaxis] * X[label == 1], axis=0)
        sum1U = np.sum(temp1)
        temp2 = U[:, k][label == 0] ** expo
        sum2 = np.sum(temp2[:, np.newaxis] * X[label == 0], axis=0)
        sum2U = np.sum(temp2)

        lcount = np.sum(label == 1)
        bcount = np.sum(label == 0)

        centers[k, :] = np.where(
            (lcount != 0) & (bcount != 0),
            wl / sum1U * sum1 + (1 - wl) / sum2U * sum2,
            np.where(
                (lcount == 0) & (bcount != 0),
                1 / sum2U * sum2,
                1 / sum1U * sum1
            )
        )
    iter = 1
    itercriterion = 1
    labels = np.zeros((cluster_num, n))
    while itercriterion > error and iter < iters:
        oldcenters = centers
        U = MystepRFCM(centers, X, expo)
        for k in range(cluster_num):

            [label, ra] = ShadowLable(U[:, k])
            labels[k] = label

            temp1 = U[:, k][label == 1] ** expo
            sum1 = np.sum(temp1[:, np.newaxis] * X[label == 1], axis=0)
            sum1U = np.sum(temp1)
            temp2 = U[:, k][label == 0] ** expo
            sum2 = np.sum(temp2[:, np.newaxis] * X[label == 0], axis=0)
            sum2U = np.sum(temp2)

            lcount = np.sum(label == 1)
            bcount = np.sum(label == 0)

            centers[k, :] = np.where(
                (lcount != 0) & (bcount != 0),
                wl / sum1U * sum1 + (1 - wl) / sum2U * sum2,
                np.where(
                    (lcount == 0) & (bcount != 0),
                    1 / sum2U * sum2,
                    1 / sum1U * sum1
                )
            )

        itercriterion = np.sqrt(np.sum((centers[0, :] - oldcenters[0, :]) ** 2))
        for k in range(1, cluster_num):
            temp = np.sqrt(np.sum((centers[k, :] - oldcenters[k, :]) ** 2))
            if itercriterion < temp:
                itercriterion = temp

        iter += 1
    return labels

