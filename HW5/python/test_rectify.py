import cv2 as cv
import numpy as np
import helper as hlp
import submission as sub
import numpy.linalg as la
import skimage.color as col
import matplotlib.pyplot as plt

# 1. Load the images and the parameters

I1 = cv.cvtColor(cv.imread('../data/im1.png'), cv.COLOR_BGR2GRAY).astype(np.float32)
I2 = cv.cvtColor(cv.imread('../data/im2.png'), cv.COLOR_BGR2GRAY).astype(np.float32)

intrinsics = np.load('../data/intrinsics.npz')
K1, K2 = intrinsics['K1'], intrinsics['K2']

extrinsics = np.load('../data/extrinsics.npz')
R1, R2 = extrinsics['R1'], extrinsics['R2']
t1, t2 = extrinsics['t1'], extrinsics['t2']

# 2. Rectify the images and save the paramters

M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = sub.rectify_pair(K1, K2, R1, R2, t1, t2)
np.savez('../data/rectify.npz', M1=M1, M2=M2, K1p=K1p, K2p=K2p, R1p=R1p, R2p=R2p, t1p=t1p, t2p=t2p)

# 3. Warp and display the result

I1, I2, bb = hlp.warpStereo(I1, I2, M1, M2)

r, c = I1.shape
I = np.zeros((r, 2*c))
I[:,:c] = I1
I[:,c:] = I2

corresp = np.load('../data/some_corresp.npz')
pts1, pts2 = corresp['pts1'][::18].T, corresp['pts2'][::18].T
pts1, pts2 = hlp._projtrans(M1, pts1), hlp._projtrans(M2, pts2)
pts2[0,:] = pts2[0,:] + c

plt.imshow(I, cmap='gray')
plt.scatter(pts1[0,:], pts1[1,:], s=60, c='r', marker='*')
plt.scatter(pts2[0,:], pts2[1,:], s=60, c='r', marker='*')
for p1, p2 in zip(pts1.T, pts2.T):
    plt.plot([p1[0],p2[0]], [p1[1],p2[1]], '-', c='b')
plt.show()
