from python.RGB2Lab import rgb2lab
from python.utils import *


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    I_lab = rgb2lab(I)
    filterResponses = []
    for i in filterBank:
        I_filt_1 = imfilter(I_lab[:,:,0],i)
        I_filt_2 = imfilter(I_lab[:,:,1],i)
        I_filt_3 = imfilter(I_lab[:,:,2],i)
        filterResponses.append(I_filt_1)
        filterResponses.append(I_filt_2)
        filterResponses.append(I_filt_3)
    # ----------------------------------------------
    
    return filterResponses
