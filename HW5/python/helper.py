"""
Homework 5
Helper Functions
"""

import cv2 as cv
import numpy as np
import scipy.optimize
import submission as sub
import numpy.linalg as la
import matplotlib.pyplot as plt


def _epipoles(E):
    U, S, V = np.linalg.svd(E)
    e1 = V[-1, :]
    U, S, V = np.linalg.svd(E.T)
    e2 = V[-1, :]

    return e1, e2


def displayEpipolarF(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, _ = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[1] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)
        plt.draw()


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))

    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))

    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=100000,
        maxfun=10000
    )

    return _singularize(f.reshape([3, 3]))


def camera2(E):
    U,S,V = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(V)
    U,S,V = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(V))<0:
        W = -W

    M2s = np.zeros([3,4,4])
    M2s[:,:,0] = np.concatenate([U.dot(W).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,1] = np.concatenate([U.dot(W).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,2] = np.concatenate([U.dot(W.T).dot(V), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)
    M2s[:,:,3] = np.concatenate([U.dot(W.T).dot(V), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1)

    return M2s


def epipolarMatchGUI(I1, I2, F):
    e1, e2 = _epipoles(F)

    sy, sx, sd = I2.shape

    f, [ax1, ax2] = plt.subplots(1, 2, figsize=(12, 9))
    ax1.imshow(I1)
    ax1.set_title('Select a point in this image')
    ax1.set_axis_off()
    ax2.imshow(I2)
    ax2.set_title('Verify that the corresponding point \n is on the epipolar line in this image')
    ax2.set_axis_off()

    while True:
        plt.sca(ax1)
        x, y = plt.ginput(1, mouse_stop=2)[0]

        xc, yc = int(x), int(y)
        v = np.array([[xc], [yc], [1]])

        l = F @ v
        s = np.sqrt(l[0]**2+l[1]**2)

        if s==0:
            error('Zero line vector in displayEpipolar')

        l = l / s
        if l[0] != 0:
            xs = 0
            xe = sx - 1
            ys = -(l[0] * xs + l[2]) / l[1]
            ye = -(l[0] * xe + l[2]) / l[1]
        else:
            ys = 0
            ye = sy - 1
            xs = -(l[1] * ys + l[2]) / l[0]
            xe = -(l[1] * ye + l[2]) / l[0]

        ax1.plot(x, y, '*', MarkerSize=6, linewidth=2)
        ax2.plot([xs, xe], [ys, ye], linewidth=2)

        # draw points
        pc = np.array([[xc, yc]])
        p2 = sub.epipolar_correspondences(I1, I2, F, pc)
        ax2.plot(p2[0,0], p2[0,1], 'ro', MarkerSize=8, linewidth=2)
        plt.draw()


def _projtrans(H, p):
    n = p.shape[1]
    p3d = np.vstack((p, np.ones((1,n))))
    h2d = H @ p3d
    p2d = h2d[:2,:] / np.vstack((h2d[2,:], h2d[2,:]))
    return p2d


def _mcbbox(s1, s2, M1, M2):
    c1 = np.array([[0,0,s1[1],s1[1]], [0,s1[0],0,s1[0]]])
    c1p = _projtrans(M1, c1)
    bb1 = [np.floor(np.amin(c1p[0,:])),
           np.floor(np.amin(c1p[1,:])),
           np.ceil(np.amax(c1p[0,:])),
           np.ceil(np.amax(c1p[1,:]))]

    c2 = np.array([[0,0,s2[1],s2[1]], [0,s2[0],0,s2[0]]])
    c2p = _projtrans(M2, c2)
    bb2 = [np.floor(np.amin(c2p[0,:])),
           np.floor(np.amin(c2p[1,:])),
           np.ceil(np.amax(c2p[0,:])),
           np.ceil(np.amax(c2p[1,:]))]

    bb = np.vstack((bb1, bb2))
    bbmin = np.amin(bb, axis=0)
    bbmax = np.amax(bb, axis=0)
    bbp = np.concatenate((bbmin[:2], bbmax[2:]))

    return bbp


def _imwarp(I, H, bb):
    #minx, miny, maxx, maxy = bb
    #dx, dy = np.arange(minx, maxx), np.arange(miny, maxy)
    #x, y = np.meshgrid(dx, dy)

    #s = x.shape
    #x, y = np.ravel(x), np.ravel(y)
    #pp = _projtrans(la.inv(H), np.vstack((x, y)))
    #x, y = pp[0][:,None].reshape(s), pp[1][:,None].reshape(s)

    s = (int(bb[2]-bb[0]), int(bb[3]-bb[1]))
    I = cv.warpPerspective(I, H, s)

    return I


def warpStereo(I1, I2, M1, M2):
    bb = _mcbbox(I1.shape, I2.shape, M1, M2)

    I1p = _imwarp(I1, M1, bb)
    I2p = _imwarp(I2, M2, bb)

    return I1p, I2p, bb
