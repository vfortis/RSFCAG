import numpy as np
from math import *
from scipy.spatial.distance import cdist

# 三叉树

def BKM3(X, ratio):
    '''
    这里平衡三叉树采取的方案是先将数据集分成1:2,然后再将2等分的数据集平均分成两份
    :param X:
    :param ratio:
    :return:
    '''
    n, m = X.shape
    cluster_num = 3
    # 初始化样本分配向量
    initlabels = np.random.randint(0, 2, n)
    labels = initlabels
    # 每个簇的样本个数
    c1 = floor(n*ratio) # 向下取整
    c2 = floor(n*(1-2*ratio))
    # 簇中心矩阵
    centers = np.zeros((cluster_num, m))

    last = np.where(labels == 0)[0]
    for _ in range(100):
        centers[:2, :] = [np.mean(X[labels == i], axis=0) for i in range(2)]
        # 2、计算分配矩阵
        # 计算样本到簇的欧几里得距离，每一行表示的是第i个样本到各个簇的距离
        distance = cdist(X, centers, metric='euclidean')
        # 取出距离矩阵的各行
        distance1 = distance[:, 0]
        distance2 = distance[:, 1]
        gap = distance1 - distance2
        idx = np.argsort(gap)
        temp = np.sort(gap)
        nn = len(np.where(temp < 0)[0])
        cp = np.clip(nn, c1, c2)
        cp = np.clip(cp, 1, n - 2)

        labels[idx[: cp]] = 0
        labels[idx[cp:]] = 1

        if np.array_equal(np.where(labels == 0)[0], last):
            break
        last = np.where(labels == 0)[0]

    n1 = len(np.where(labels != 0)[0])
    labels[np.where(labels != 0)[0]] = np.random.randint(1, 3, n1)
    c1 = floor(n1*0.5)
    c2 = floor(n1*(1-0.5))

    last = np.where(labels == 1)[0]
    for _ in range(100):
        centers[1:, :] = [np.mean(X[labels == i], axis=0) for i in range(1, 3)]
        # 2、计算分配矩阵
        # 计算样本到簇的欧几里得距离，每一列表示的是每个样本到各个簇的距离
        distance = cdist(X, centers, metric='euclidean')
        distance[np.where(labels == 0)[0]] = 0 # 将第一次二分的结果去除在外
        # 取出距离矩阵的各行
        distance1 = distance[:, 1]
        distance2 = distance[:, 2]
        gap = distance1 - distance2
        idx = np.argsort(gap)
        temp = np.sort(gap[np.where(labels != 0)[0]])
        nn = len(np.where(temp < 0)[0])
        cp = np.clip(nn, c1, c2)
        cp = np.clip(cp, 1, n1 - 1)

        labels[[i for i in idx if i in np.where(labels != 0)[0]][:cp]] = 1
        labels[[i for i in idx if i in np.where(labels != 0)[0]][cp:]] = 2
        if np.array_equal(np.where(labels == 1)[0], last):
            break
        last = np.where(labels == 1)[0]


    return centers, labels








