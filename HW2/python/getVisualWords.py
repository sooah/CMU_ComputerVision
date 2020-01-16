import numpy as np
from scipy.spatial.distance import cdist
from python.extractFilterResponses import extract_filter_responses


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    filt_img = extract_filter_responses(I, filterBank)

    height = I.shape[0]
    width = I.shape[1]

    wordMap = np.zeros((I.shape[0], I.shape[1]))
    for y in range(height):
        for x in range(width):
            filt_vec = np.asarray([filt_img[n][y][x] for n in range(len(filt_img))])
            dist = cdist(dictionary, [filt_vec], metric = 'euclidean')
            min_idx = np.argmin(dist)
            wordMap[y][x] = min_idx
    # ----------------------------------------------

    return wordMap

