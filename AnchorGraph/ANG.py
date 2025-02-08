import numpy as np
from scipy.spatial.distance import cdist
import scipy.sparse as sp

def ANG(X, anchor, k):
    # 每一行表示的是第i个样本到各个簇的距离
    distance = cdist(X, anchor, metric='euclidean')
    # 按列排序的元素在原始数组中的行索引
    idx = np.argsort(distance, axis=1)
    idx1 = idx[:, 0:k+1]
    n, _ = X.shape
    numanchor, _ = anchor.shape
    A = np.zeros((n, numanchor))
    for i in range(n):
        id = idx1[i, 0:k+1]
        di = distance[i, id]
        A[i, id] = (di[k]-di)/(k*di[k]-np.sum(di[0:k]))  # eps

    # 使用压缩稀疏行（Compressed Sparse Row）格式存储稀疏矩阵
    # A = sp.csr_matrix(A)
    return A




