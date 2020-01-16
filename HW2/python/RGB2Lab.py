import cv2 as cv
import numpy as np


def rgb2lab(img_rgb):
    if img_rgb.max() > 1.0:
        img_rgb = img_rgb / 255.0

    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]

    M, N = img_rgb.shape[:2]
    RGB = np.stack([R.ravel(), G.ravel(), B.ravel()])

    # RGB to XYZ
    MAT = np.array([[0.412453, 0.357580, 0.180423],
                    [0.212671, 0.715160, 0.072169],
                    [0.019334, 0.119193, 0.950227]])
    XYZ = MAT.dot(RGB)

    X = XYZ[0, :] / 0.950456
    Y = XYZ[1, :]
    Z = XYZ[2, :] / 1.088754

    T = 0.008856
    XT = X > T
    YT = Y > T
    ZT = Z > T

    fX = XT * (X **(1 / 3)) + (1 - XT) * (7.787 * X + 16 / 116)

    # Compute L
    Y3 = Y ** (1 / 3)
    fY = YT * Y3 + (1 - YT) * (7.787 * Y + 16 / 116)
    L = YT * (116 * Y3 - 16.0) + (1 - YT) * (903.3 * Y)

    fZ = ZT * (Z ** (1 / 3)) + (1 - ZT) * (7.787 * Z + 16 / 116)

    # Compute a and b
    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    Lab = np.stack([L.reshape([M, N]), a.reshape([M, N]), b.reshape([M, N])], axis=2)
    return Lab

