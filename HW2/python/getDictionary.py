import numpy as np
import cv2 as cv
from python.createFilterBank import create_filterbank
from python.extractFilterResponses import extract_filter_responses
from python.getRandomPoints import get_random_points
from python.getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    point_count = 0

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        filt_response = extract_filter_responses(image, filterBank)

        if method == 'Random':
            points = get_random_points(image, alpha)
            for idx in range(alpha):
                y = points[0][0][idx]
                x = points[0][1][idx]
                for n in range(len(filt_response)):
                    get_pos_value = filt_response[n][y, x]
                    pixelResponses[point_count, n] = get_pos_value
                point_count = point_count + 1

        elif method == 'Harris':
            points = get_harris_points(image, alpha, k = 0.05)
            for y, x in points:
                for n in range(len(filt_response)):
                    get_pos_value = filt_response[n][y, x]
                    pixelResponses[point_count, n] = get_pos_value
                point_count = point_count + 1
        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary