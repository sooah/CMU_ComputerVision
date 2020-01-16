import numpy as np
from scipy import ndimage


def imfilter(I, h):
    I_f = ndimage.filters.correlate(I, h, mode='constant')
    return I_f


def fspecial_gaussian(size, sigma=0.5):
    m = (size-1) / 2
    y, x = np.ogrid[-m:m+1, -m:m+1]
    h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def fspecial_log(size, std):
   siz = int((size-1)/2)
   x = y = np.linspace(-siz, siz, 2*siz+1)
   x, y = np.meshgrid(x, y)
   arg = -(x**2 + y**2) / (2*std**2)
   h = np.exp(arg)
   h[h < np.finfo(h.dtype).eps*h.max()] = 0
   h = h/h.sum() if h.sum() != 0 else h
   h1 = h*(x**2 + y**2 - 2*std**2) / (std**4)
   return h1 - h1.mean()


def chi2dist(X, Y):
    s = X + Y
    d = Y - X
    d = (d ** 2 / (s + 1e-10)).sum() / 2.0
    return d