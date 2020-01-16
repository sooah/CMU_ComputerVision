import numpy as np

from python.utils import chi2dist
from scipy.spatial.distance import euclidean

def get_image_distance(hist1, hist2, method):
    if method == 'euclidean':
        dist = euclidean(hist1, hist2)
    elif method == 'chi2':
        dist = chi2dist(hist1, hist2)
    return dist

def getImageDistance(hist1, histSet, method):
    dist = []
    for trainhist in histSet:
        trainhist = trainhist.reshape(-1, 1)
        if method == 'euclidean':
            dist_ = euclidean(hist1, trainhist)
        elif method == 'chi2':
            dist_ = chi2dist(hist1, trainhist)
        dist.append(dist_)
    return dist