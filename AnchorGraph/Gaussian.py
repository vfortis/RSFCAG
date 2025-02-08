import numpy as np


def gaussian_kernel(x, y, sigma=1.0):
    """
    高斯核函数计算两个向量之间的相似度。

    参数：
        x: 第一个向量
        y: 第二个向量
        sigma: 高斯核函数的标准差，默认为 1.0

    返回值：
        相似度得分
    """
    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


def compute_adjacency_matrix(node_features, sigma=1.0):
    """
    使用高斯核函数计算邻接矩阵。

    参数：
        node_features: 节点特征的矩阵，每行是一个节点的特征向量
        sigma: 高斯核函数的标准差，默认为 1.0

    返回值：
        邻接矩阵
    """
    num_nodes = len(node_features)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            adjacency_matrix[i, j] = gaussian_kernel(node_features[i], node_features[j], sigma)
    return adjacency_matrix


