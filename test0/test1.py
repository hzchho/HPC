import time
import random
M,N,K=map(int,input().split())
#初始化
A=[[random.uniform(0,1) for j in range(N)] for i in range(M)]
B=[[random.uniform(0,1) for j in range(K)] for i in range(N)]
C=[[0 for j in range(K)] for i in range(M)]
#开始时间
time1=time.time()
#矩阵运算
for i in range(M):
    for l in range(N):
        for j in range(K):
            C[i][j]+=A[i][l]*B[l][j]
#结束时间
time2=time.time()
print("Used time:",(time2-time1).__round__(6),"s")