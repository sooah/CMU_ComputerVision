from utils import *


def create_filterbank():
    # Code to generate reasonable filter bank

    gaussianScales = [1, 2, 4, 8, np.sqrt(2)*8]
    logScales      = [1, 2, 4, 8, np.sqrt(2)*8]
    dxScales       = [1, 2, 4, 8, np.sqrt(2)*8]
    dyScales       = [1, 2, 4, 8, np.sqrt(2)*8]

    filterBank = []

    for scale in gaussianScales:
        filter = fspecial_gaussian(2*np.ceil(scale*2.5)+1, scale)
        filterBank.append(filter)

    for scale in logScales:
        filter = fspecial_log(2*np.ceil(scale*2.5)+1, scale)
        filterBank.append(filter)

    for scale in dxScales:
        filter0 = fspecial_gaussian(2 * np.ceil(scale * 2.5) + 1, scale)
        filter = imfilter(filter0, np.array([[-1, 0, 1]]))
        filterBank.append(filter)

    for scale in dyScales:
        filter0 = fspecial_gaussian(2 * np.ceil(scale * 2.5) + 1, scale)
        filter = imfilter(filter0, np.array([[-1], [0], [1]]))
        filterBank.append(filter)

    return filterBank
