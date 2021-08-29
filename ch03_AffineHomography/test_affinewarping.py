from PIL import Image
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

import warp

def test_1():
    # アフィン変換

    img = np.array(Image.open("2007_000333.jpg").convert('L'))
    H = np.array([[1.4, 0.05, -100],
                  [0.05, 1.5, -100],
                  [0,0,1]])
    img2 = ndimage.affine_transform(img, H[:2,:2], (H[0,2], H[1,2]))

    plt.gray()
    plt.imshow(img2)
    plt.show()


def test_2():
    # 指定した対応点からアフィン変換行列を作成して、適用

    img1 = np.array(Image.open("2007_000333.jpg").convert('L'))
    img2 = np.array(Image.open("2007_000033.jpg").convert('L'))

    # 点を設定する
    tp = np.array([[  0, 100, 100, 0],    # y
                   [300, 300, 470, 470],  # x
                   [  1,   1,   1,   1]]) # 1

    # アフィン変換
    img3 = warp.image_in_image(img1, img2, tp)

    plt.gray()
    plt.imshow(img3)
    plt.show()


def test_3():
    """アフィン変換は、画像の四隅の4つの点すべてを目標の位置に変換することができない。
    (完全な射影変換なら可能だが、)
    """

if __name__ == "__main__":
    # test_1()
    test_2()