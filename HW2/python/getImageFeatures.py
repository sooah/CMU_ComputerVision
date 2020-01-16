import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    # for each in wordMap:
    #     hist = np.histogram(each, bins = np.arange(dictionarySize))
    # total_hist = []
    # n = 0
    # for wordmap in wordMap:
    #     hist = np.histogram(wordmap, bins = np.arange(dictionarySize))
    #     if n == 0:
    #         total_hist = hist[0]/np.sum(hist[0])
    #     else:
    #         total_hist = np.vstack((total_hist,hist[0]))
    #     n = n+1
    #
    # h = np.sum(total_hist,axis=0)
    n = 0
    for wordmap in wordMap:
        hist = np.histogram(wordmap, bins = np.arange(dictionarySize))
        if n == 0:
            h = hist[0]/np.sum(hist[0])
        else:
            h = np.vstack((h, hist[0]/np.sum(hist[0])))
        n = n+1
    # ----------------------------------------------
    
    return h
