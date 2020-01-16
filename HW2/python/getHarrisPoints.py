import numpy as np
import cv2 as cv
from scipy import ndimage
from python.utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    I_x = imfilter(I, sobel_x)
    I_y = imfilter(I, sobel_y)
    I_xx = I_x**2
    I_xy = I_y*I_x
    I_yy = I_y**2

    height = I.shape[0]
    width = I.shape[1]
    harris_response = []
    offset = (np.floor(sobel_x.shape[0]/2)).astype(int)

    # Harris response calculation
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            S_xx = np.sum(I_xx[y-offset:y+1+offset, x-offset:x+1+offset])
            S_yy = np.sum(I_yy[y-offset:y+1+offset, x-offset:x+1+offset])
            S_xy = np.sum(I_xy[y-offset:y+1+offset, x-offset:x+1+offset])

            # Find determinant and trace, use to get corner response
            det = (S_xx*S_yy) - (S_xy**2)
            trace = S_xx + S_yy
            r = det - k*(trace**2)

            harris_response.append([y, x, r])

    harris_response_sorted = sorted(harris_response, key = lambda x:x[2], reverse = True)

    # Find edges and corners using R
    # edge : r<0 / corner : r>0 / flat : r=0
    points = []
    for response in harris_response_sorted[0:alpha]:
        y, x, r = response
        if r > 0:
            points.append([y, x])
    # ----------------------------------------------
    
    return points

