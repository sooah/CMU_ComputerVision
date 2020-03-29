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

rectify = np.load('../data/rectify.npz')
M1, M2 = rectify['M1'], rectify['M2']
K1p, K2p = rectify['K1p'], rectify['K2p']
R1p, R2p = rectify['R1p'], rectify['R2p']
t1p, t2p = rectify['t1p'], rectify['t2p']

# 2. Get disparity and depth maps

max_disp, win_size = 20, 3
dispM = sub.get_disparity(I1, I2, max_disp, win_size)
depthM = sub.get_depth(dispM, K1p, K2p, R1p, R2p, t1p, t2p)

# 3. Display disparity and depth maps

dispI = dispM * (I1>40)
depthI = depthM * (I1>40)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(dispI, cmap='gray')
ax1.set_title('Disparity Image')
ax1.set_axis_off()
ax2.imshow(depthI, cmap='gray')
ax2.set_title('Depth Image')
ax2.set_axis_off()
plt.tight_layout()
plt.show()
