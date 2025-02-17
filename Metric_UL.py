import numpy as np

def Metric_UL(r, t, ko):
    if r.ndim == 1:
        r = r.reshape(1, -1)
    if t.ndim == 1:
        t = t.reshape(1, -1)
    
    M, N = r.shape[1], t.shape[1]
    
    d = (np.tile(r.reshape(-1, 1), (1, N)) - np.tile(t, (M, 1))) ** 2
    d = d * ko
    D = np.zeros(d.shape)
    D[0, 0] = d[0, 0]
    
    for m in range(1, M):
        D[m, 0] = d[m, 0] + D[m-1, 0]
    
    for n in range(1, N):
        D[0, n] = d[0, n] + D[0, n-1]
    
    for m in range(1, M):
        for n in range(1, N):
            D[m, n] = d[m, n] + min(D[m-1, n], min(D[m-1, n-1], D[m, n-1]))
    
    return D[-1, -1]
