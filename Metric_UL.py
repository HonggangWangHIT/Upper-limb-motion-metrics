import numpy as np

def Metric_UL(r, t, ko):
    # 确保 r 和 t 是二维数组
    if r.ndim == 1:
        r = r.reshape(1, -1)  # 将一维数组转换为行向量
    if t.ndim == 1:
        t = t.reshape(1, -1)  # 将一维数组转换为行向量
    
    M, N = r.shape[1], t.shape[1]
    
    # 计算 d 矩阵
    # d = (np.tile(r.T, (1, N)) - np.tile(t, (M, 1))) ** 2
    # d = d * ko
    d = (np.tile(r.reshape(-1, 1), (1, N)) - np.tile(t, (M, 1))) ** 2
    d = d * ko
    # 初始化 D 矩阵
    D = np.zeros(d.shape)
    D[0, 0] = d[0, 0]
    
    # 填充 D 矩阵的第一列
    for m in range(1, M):
        D[m, 0] = d[m, 0] + D[m-1, 0]
    
    # 填充 D 矩阵的第一行
    for n in range(1, N):
        D[0, n] = d[0, n] + D[0, n-1]
    
    # 填充 D 矩阵的其余部分
    for m in range(1, M):
        for n in range(1, N):
            D[m, n] = d[m, n] + min(D[m-1, n], min(D[m-1, n-1], D[m, n-1]))
    
    # 返回最终的 Dist
    return D[-1, -1]