from FCM import FCM
from RJSFCAG import RJSFCAG
from utils.evaluation import evaluate
from AnchorGraph.PNG import PNG
from AnchorSelection.TKHK import TKHK
import scipy.io as scio
import numpy as np
import logging
import openpyxl

# 数据准备
matdata = scio.loadmat(r'data/COIL100.mat')
X = matdata['fea']
gnd = matdata['gnd'].squeeze()
cluster_num = len(np.unique(gnd))

# 参数设置
iters = 50
error = 0.0001
circ = 1
AccRJSFCAG = 0
rRegion = [5 for i in range(5, 45, 5)]
GammaRegion = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]
MuRegion = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3]
EntaRegion = [10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2]

# rRegion = [5]
# GammaRegion = [1]
# MuRegion = [1]
# EntaRegion = [1]

# 初始化：FCM clustering
VFCM, UFCM, _ = FCM(X, cluster_num, iters, error)

# 生成锚点
numanchor = 5  # 层数
anchor = TKHK(X, numanchor)
anchornumber = anchor.shape[0]  # 锚点数量

# 相似矩阵
k = 5 # 相似图稀疏控制
A = PNG(X, anchor, k)

for r in rRegion:
    for gi in range(len(GammaRegion)):
        gamma = GammaRegion[gi]
        for mi in range(len(MuRegion)):
            mu = MuRegion[mi]
            for eni in range(len(EntaRegion)):
                enta = EntaRegion[eni]

                P, V, U, step = RJSFCAG(anchor, A, UFCM, r, cluster_num, gamma, mu, enta, iters, error)

                URJSFCAG = U
                grpsRJSFCAG = np.argmax(URJSFCAG, axis=1)
                nmi, ari, f, acc = evaluate(gnd, grpsRJSFCAG)

                if AccRJSFCAG < acc:
                    AccRJSFCAG = acc
                    MiRJSFCAG = nmi
                    AriRJSFCAG = ari
                    F_scoreRJSFCAG = f
                    rRJSFCAG = r
                    GammaRJSFCAG = gamma
                    MuRJSFCAG = mu
                    EntaRJSFCAG = enta
                    StepRJSFCAG = step

                    Popt = P
                    Vopt = V
                    Uopt = U

                circ += 1
                print(f'ID: {circ}, r = {r}, gamma = {gamma}, mu = {mu}, enta = {enta}, ACC = {acc}, NMI = {nmi}, ARI = {ari}, F_score = {f}, Step = {step}')

print(f'RJSFCAG Clustering: r = {rRJSFCAG}, gamma = {GammaRJSFCAG}, mu = {MuRJSFCAG}, enta = {EntaRJSFCAG}, ACC = {AccRJSFCAG}, NMI = {MiRJSFCAG}, ARI = {AriRJSFCAG}, F_score = {F_scoreRJSFCAG}, Step = {StepRJSFCAG}')


