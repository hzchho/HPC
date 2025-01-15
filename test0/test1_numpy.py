import numpy as np
import time
M,N,K=map(int,input().split())
A=np.random.rand(M,N)
B=np.random.rand(N,K)
C=np.random.rand(M,K)
time1=time.time()
C=A@B
time2=time.time()
print("Used time:",(time2-time1).__round__(6),"s")