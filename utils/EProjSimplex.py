import numpy as np

def EProjSimplex(v):
    n = len(v)
    ft = 1
    v0 = v - np.mean(v) + 1/n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-10:
            v1 = lambda_m - v0
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = npos/n - 1
            f = np.sum(v1[posidx])/n - lambda_m
            lambda_m = lambda_m - f/g
            ft += 1
            if ft > 100:
                x = np.maximum(-v1, 0)
                break
        x = np.maximum(-v1, 0)
    else:
        x = v0
    return x
