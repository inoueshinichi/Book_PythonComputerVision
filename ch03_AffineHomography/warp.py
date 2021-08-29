from homography import Haffine_from_points

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def image_in_image(img1, img2, tp):
    """四隅をできるだけtpに近づけるアフィン変換を使って
       img1をimg2に埋め込む.
       tpは同次座標で、左上から逆時計回りにとる.
    """

    # 元の座標
    m, n = img1.shape[:2]

    # 縦ベクトルが同次座標
    fp = np.array([[0, m, m, 0], # y
                   [0, 0, n, n], # x
                   [1, 1, 1 ,1]])# 1

    # アフィン変換を計算し、適用する
    H = Haffine_from_points(tp, fp)
    img1_t = ndimage.affine_transform(img1, H[:2, :2], (H[0, 2], H[1, 2]), img2.shape[:2])
    plt.imshow(img1_t)
    plt.show()

    alpha = (img1_t > 0) # アフィン変換の領域外の画素値は0になっていると仮定

    return (1 - alpha) * img2 + alpha * img1_t