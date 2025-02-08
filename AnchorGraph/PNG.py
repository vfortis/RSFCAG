import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp

def PNG(X, anchor, k):
    # 每一行表示的是第i个样本到各个簇的距离
    distance = cdist(X, anchor, metric='euclidean')
    # 按列排序的元素在原始数组中的行索引
    idx = np.argsort(distance, axis=1)
    idx1 = idx[:, :k]
    n, _ = X.shape
    numanchor, _ = anchor.shape
    A = np.zeros((n, numanchor))
    for i in range(n):
        id = idx1[i]
        di = distance[i, id]
        thetai = distance[i, id[-1]]
        A[i, id] = np.exp(-di/thetai)  # eps

    # 使用压缩稀疏行（Compressed Sparse Row）格式存储稀疏矩阵
    # A = sp.csr_matrix(A)
    return A