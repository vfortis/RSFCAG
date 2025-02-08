from utils.EProjSimplex import EProjSimplex
from utils.EuDist2 import EuDist2
import numpy as np
import scipy.sparse as sp

def RJSFCAG(anchor, B, initU, r, cluster_num, Pgamma, Pmu, Penta, iters, error):
    X = B.dot(anchor)  # 稠密矩阵----更高阶的特征
    n = X.shape[0]
    m = X.shape[1]

    expo = 2
    mf = np.power(initU, expo)
    V = np.dot(mf.T, X) / np.reshape(np.sum(mf, axis=0), (cluster_num, 1))

    U = initU

    P = np.eye(m, r)

    iter = 0
    converged = False
    obj = []


    while not converged and iter <= iters:
        iter += 1
        oldV = V

        # Update U
        XP = np.dot(X, P)
        H = EuDist2(XP, np.dot(V, P))

        vi = -1 * H / (2 * Pgamma)
        U = np.apply_along_axis(EProjSimplex, 1, vi)

        # Update V
        G = U / (2 * H + np.finfo(float).eps)
        temp = np.sum(G, axis=0)
        if len(np.where(temp == 0)[0]) > 0:
            V = np.dot(G.T, X) / np.reshape(temp + np.finfo(float).eps, (cluster_num, 1))
        else:
            V = np.dot(G.T, X) / np.reshape(temp, (cluster_num, 1))

        # Update P
        H = EuDist2(XP, np.dot(V, P))   # 此时V变化了
        G = U / (2 * H + np.finfo(float).eps)  # H变化
        G1 = np.diag(np.sum(G, axis=1))
        G2 = np.diag(np.sum(G, axis=0))

        mm = np.dot(B.T, B)
        delta = (1.0 / np.sum(B, axis=0)).squeeze()
        delta = np.diag(delta)
        Ls = mm - np.dot(np.dot(mm, delta), mm)

        D = np.sqrt(np.sum(np.power(P, 2), axis=1))
        D = np.diag(1 / (2 * D + np.finfo(float).eps))

        AA = np.dot(X.T, np.dot(G1, X)) - (np.dot(X.T, np.dot(G, V)) + np.dot(V.T, np.dot(G.T, X))) + \
             np.dot(V.T, np.dot(G2, V)) + Pmu * np.dot(anchor.T, np.dot(Ls, anchor)) + Penta * D

        if np.any(np.isnan(AA)) or np.any(np.isinf(AA)):
            # 对数组进行处理，如替换非数值为零或其他操作
            AA[np.isnan(AA)] = 0.0
            AA[np.isinf(AA)] = np.finfo(AA.dtype).max

        eigvalue, eigvector = np.linalg.eig(AA)
        # 对特征值和特征向量进行排序
        sort_indice = np.argsort(eigvalue)
        eigvalue = eigvalue[sort_indice]
        eigvector = eigvector[:, sort_indice]

        if r < len(eigvalue):
            eigvector = eigvector[:, :r]

        P = eigvector  # dim_n*r

        P = P / np.linalg.norm(P, axis=0)

        # XP = np.dot(X, P)
        # VP = np.dot(V, P)
        # anchorP = np.dot(anchor, P)
        # # 计算 obj
        # obj1 = np.trace(np.dot(np.dot(XP.T, G1), XP)) - 2 * np.trace(
        #     np.dot(np.dot(XP.T, G), VP)) + np.trace(np.dot(np.dot(VP.T, G2), VP))
        # obj2 = Pgamma * np.linalg.norm(U, 2)  # 这里应该为 F 范数
        # obj3 = Pmu * np.trace(np.dot(np.dot(anchorP.T, Ls), anchorP))
        # obj4 = Penta * np.trace(np.dot(np.dot(P.T, D), P))
        # obj5 = obj1 + obj2 + obj3 + obj4
        # obj.append(obj5)

        ErrorV= np.linalg.norm(V - oldV, axis=1)
        criterion = np.max(ErrorV)
        converged = criterion < error
        print(iter)
    return P, V, U, iter



