import numpy as np
from scipy.linalg import det
from scipy.ndimage import uniform_filter
from Metric_UL import Metric_UL

def UP_Ldw(in_data1, in_data2):
    HD1X, HD1Y, HD1Z = in_data1[:, 6] * 0.01, in_data1[:, 7] * 0.01, in_data1[:, 8] * 0.01
    EL1X, EL1Y, EL1Z = in_data1[:, 3] * 0.01, in_data1[:, 4] * 0.01, in_data1[:, 5] * 0.01
    GH1X, GH1Y, GH1Z = in_data1[:, 0] * 0.01, in_data1[:, 1] * 0.01, in_data1[:, 2] * 0.01

    HD2X, HD2Y, HD2Z = in_data2[:, 6] * 0.01, in_data2[:, 7] * 0.01, in_data2[:, 8] * 0.01
    EL2X, EL2Y, EL2Z = in_data2[:, 3] * 0.01, in_data2[:, 4] * 0.01, in_data2[:, 5] * 0.01
    GH2X, GH2Y, GH2Z = in_data2[:, 0] * 0.01, in_data2[:, 1] * 0.01, in_data2[:, 2] * 0.01

    aj, bj = len(HD1X), len(HD2X)

    HD1X, HD1Y, HD1Z = uniform_filter(HD1X, size=5, mode='nearest'), uniform_filter(HD1Y, size=5, mode='nearest'), uniform_filter(HD1Z, size=5, mode='nearest')
    EL1X, EL1Y, EL1Z = uniform_filter(EL1X, size=5, mode='nearest'), uniform_filter(EL1Y, size=5, mode='nearest'), uniform_filter(EL1Z, size=5, mode='nearest')
    GH1X, GH1Y, GH1Z = uniform_filter(GH1X, size=5, mode='nearest'), uniform_filter(GH1Y, size=5, mode='nearest'), uniform_filter(GH1Z, size=5, mode='nearest')

    HD2X, HD2Y, HD2Z = uniform_filter(HD2X, size=5, mode='nearest'), uniform_filter(HD2Y, size=5, mode='nearest'), uniform_filter(HD2Z, size=5, mode='nearest')
    EL2X, EL2Y, EL2Z = uniform_filter(EL2X, size=5, mode='nearest'), uniform_filter(EL2Y, size=5, mode='nearest'), uniform_filter(EL2Z, size=5, mode='nearest')
    GH2X, GH2Y, GH2Z = uniform_filter(GH2X, size=5, mode='nearest'), uniform_filter(GH2Y, size=5, mode='nearest'), uniform_filter(GH2Z, size=5, mode='nearest')

    Tborg1 = np.array([GH1X[0], GH1Y[0], GH1Z[0]])
    Tborg2 = np.array([GH2X[0], GH2Y[0], GH2Z[0]])

    GH1X, GH1Y, GH1Z = GH1X - Tborg1[0], GH1Y - Tborg1[1], GH1Z - Tborg1[2]
    EL1X, EL1Y, EL1Z = EL1X - Tborg1[0], EL1Y - Tborg1[1], EL1Z - Tborg1[2]
    HD1X, HD1Y, HD1Z = HD1X - Tborg1[0], HD1Y - Tborg1[1], HD1Z - Tborg1[2]

    GH2X, GH2Y, GH2Z = GH2X - Tborg2[0], GH2Y - Tborg2[1], GH2Z - Tborg2[2]
    EL2X, EL2Y, EL2Z = EL2X - Tborg2[0], EL2Y - Tborg2[1], EL2Z - Tborg2[2]
    HD2X, HD2Y, HD2Z = HD2X - Tborg2[0], HD2Y - Tborg2[1], HD2Z - Tborg2[2]

    a1, a2, a3 = np.zeros(aj), np.zeros(aj), np.zeros(aj)
    a, Lah, Laf = np.zeros(aj), np.zeros(aj), np.zeros(aj)
    b1, b2, b3 = np.zeros(bj), np.zeros(bj), np.zeros(bj)
    b, Lbh, Lbf = np.zeros(bj), np.zeros(bj), np.zeros(bj)

    for i in range(aj):
        A = np.array([[HD1X[i], HD1Y[i], 1], [EL1X[i], EL1Y[i], 1], [GH1X[i], GH1Y[i], 1]])
        B = np.array([[HD1X[i], HD1Z[i], 1], [EL1X[i], EL1Z[i], 1], [GH1X[i], GH1Z[i], 1]])
        C = np.array([[HD1Y[i], HD1Z[i], 1], [EL1Y[i], EL1Z[i], 1], [GH1Y[i], GH1Z[i], 1]])

        a1[i], a2[i], a3[i] = 0.5 * det(A), 0.5 * det(B), 0.5 * det(C)
        a[i] = np.sqrt(a1[i]**2 + a2[i]**2 + a3[i]**2)
        Lah[i] = np.sqrt((GH1X[i] - EL1X[i])**2 + (GH1Y[i] - EL1Y[i])**2 + (GH1Z[i] - EL1Z[i])**2)
        Laf[i] = np.sqrt((HD1X[i] - EL1X[i])**2 + (HD1Y[i] - EL1Y[i])**2 + (HD1Z[i] - EL1Z[i])**2)

    Laah, Laaf = np.mean(Lah), np.mean(Laf)

    for i in range(bj):
        D = np.array([[HD2X[i], HD2Y[i], 1], [EL2X[i], EL2Y[i], 1], [GH2X[i], GH2Y[i], 1]])
        E = np.array([[HD2X[i], HD2Z[i], 1], [EL2X[i], EL2Z[i], 1], [GH2X[i], GH2Z[i], 1]])
        F = np.array([[HD2Y[i], HD2Z[i], 1], [EL2Y[i], EL2Z[i], 1], [GH2Y[i], GH2Z[i], 1]])

        b1[i], b2[i], b3[i] = 0.5 * det(D), 0.5 * det(E), 0.5 * det(F)
        b[i] = np.sqrt(b1[i]**2 + b2[i]**2 + b3[i]**2)
        Lbh[i] = np.sqrt((GH2X[i] - EL2X[i])**2 + (GH2Y[i] - EL2Y[i])**2 + (GH2Z[i] - EL2Z[i])**2)
        Lbf[i] = np.sqrt((HD2X[i] - EL2X[i])**2 + (HD2Y[i] - EL2Y[i])**2 + (HD2Z[i] - EL2Z[i])**2)

    Lbbh, Lbbf = np.mean(Lbh), np.mean(Lbf)

    koh, kof = Laah / Lbbh, Laaf / Lbbf
    ko = 1 / koh * kof

    b1, b2, b3 = b1 * ko, b2 * ko, b3 * ko
    a1, a2, a3 = a1 / ko, a2 / ko, a3 / ko

    b1, b2, b3 = b1 + (a1[0] - b1[0]), b2 + (a2[0] - b2[0]), b3 + (a3[0] - b3[0])

    Dist1 = Metric_UL(a1, b1, ko)
    Dist2 = Metric_UL(a2, b2, ko)
    Dist3 = Metric_UL(a3, b3, ko)
    Dist = Dist1 + Dist2 + Dist3

    return Dist