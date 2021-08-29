import numpy as np

def normalize(points):
    """同次座標系の点の集合を, 最後のrow=1になるように正規化する

    Args:
        points (numpy): 行(x,y,w), 列(データ点) = (dim,N)
    """
    for row in points:
        row /= points[-1]
    return points


def make_homog(points):
    """点の集合(dim,N)を同次座標系に変換する

    Args:
        points (numpy): (dim,N)

    Returns:
        [numpy]: (dim+1,N)
    """
    return np.vstack((points, np.ones((1, points.shape[1]))))


def H_from_points(fp, tp):
    """線形なDLT法を使って、fpをtpに対応づけるホモグラフィー行列Hを求める.
       対応点は平均0, 標準偏差1に調整される.

    Args:
        fp (numpy): (dim+1, N)
        tp (numpy): (dim+1, N)

    Raises:
        RuntimeError: [description]

    Returns:
        H: (3, 3)
    """

    if fp.shape != tp.shape:
        raise RuntimeError("Number of points don't match.")

    # 点を調整する(数値計算上重要): 点群は平均0, 標準偏差1に正規化する
    # 開始点
    m = np.mean(fp[:2], axis=1) # (dim, 1)
    max_std = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / max_std, 1 / max_std, 1]) # (3, 3)
    C1[0, 2] = -m[0] / max_std
    C1[1, 2] = -m[1] / max_std
    fp = np.dot(C1, fp) # (3, 3) @ (3, N) = (3, N)

    # 対応点
    m = np.mean(tp[:2], axis=1)
    max_std = np.max(np.std(tp[:2], axis=1)) + 1e-9
    C2 = np.diag([1 / max_std, 1 / max_std, 1]) # (3, 3)
    C2[0, 2] = -m[0] / max_std
    C2[1, 2] = -m[1] / max_std

    # 線形法(DLT: Direct Linear Transformation)
    # のための行列を作る. 対応ごとに2つの行になる.
    nbr_correspondences = fp.shape[1]
    A = np.zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):
        A[2*i] = [ -fp[0,i], -fp[1,i], -1, 0, 0, 0, fp[0,i]*tp[0,i], fp[1,i]*tp[0,i], tp[0,i] ]
        A[2*i+1] = [ 0, 0, 0, -fp[0,i], -fp[1,i], -1, fp[0,i]*tp[1,i], fp[1,i]*tp[1,i], tp[1,i] ]

    U, S, V = np.linalg.svd(A) # 特異値分解
    H = V[8].reshape((3,3)) # 最小二乗法の解は、SVDの行列Vの最後の行として求まる

    # 調整を元に戻す
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    # 正規化して返す
    return H / H[2, 2]


def Haffine_from_points(fp, tp):
    """fpをtpに変換するアフィン変換行列Hを求める
    文献[13]の130ページに記載されている方法で実装

    Args:
        fp ([type]): [description]
        tp ([type]): [description]
    """

    if fp.shape != tp.shape:
        raise RuntimeError("number of points do not match.")

    # 点を調整する(数値計算上重要): 点群は平均0, 標準偏差1に正規化する
    # 開始点
    m = np.mean(fp[:2], axis=1)
    max_std = np.max(np.std(fp[:2], axis=1)) + 1e-9
    C1 = np.diag([1 / max_std, 1 / max_std, 1])
    C1[0, 2] = -m[0] / max_std
    C1[1, 2] = -m[1] / max_std
    fp_cond = np.dot(C1, fp)

    # 対応点
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy() # 2つの点群で同じ拡大率を使う
    C2[0, 2] = -m[0] / max_std
    C2[1, 2] = -m[1] / max_std
    tp_cond = np.dot(C2, tp)

    # 平均０になるように調整する. 平行移動はなくなる.
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    # Hartley-Zisserman(第2版)p.130に基づき、行列B, Cを求める
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2,1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # 調整を元に戻す
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2,2]





