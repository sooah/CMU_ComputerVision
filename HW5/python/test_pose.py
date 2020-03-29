import numpy as np
import submission as sub
import numpy.linalg as la

# 1. Generate random camera matrix

K = np.array([[1,0,100], [0,1,100], [0,0,1]])
R, _,_ = la.svd(np.random.randn(3,3))
if la.det(R) < 0: R = -R
t = np.vstack((np.random.randn(2,1), 1))

P = K @ np.hstack((R, t))

# 2. Generate random 2D and 3D points

N = 100

X = np.random.randn(N,3)
x = P @ np.hstack((X, np.ones((N,1)))).T
x = x[:2,:].T / np.vstack((x[2,:], x[2,:])).T

# 3. Test pose estimation with clean points

Pc = sub.estimate_pose(x, X)

xp = Pc @ np.hstack((X, np.ones((N,1)))).T
xp = xp[:2,:].T / np.vstack((xp[2,:], xp[2,:])).T

print('Reprojection Error with clean 2D points:', la.norm(x-xp))
print('Pose Error with clean 2D points:', la.norm((Pc/Pc[-1,-1])-(P/P[-1,-1])))

# 4. Test pose estimation with noisy points

x = x + np.random.rand(x.shape[0], x.shape[1])
Pn = sub.estimate_pose(x, X)

xp = Pn @ np.hstack((X, np.ones((N,1)))).T
xp = xp[:2,:].T / np.vstack((xp[2,:], xp[2,:])).T

print('Reprojection Error with noisy 2D points:', la.norm(x-xp))
print('Pose Error with noisy 2D points:', la.norm((Pn/Pn[-1,-1])-(P/P[-1,-1])))
