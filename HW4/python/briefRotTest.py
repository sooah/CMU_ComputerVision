import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
from helper import plotMatches

import matplotlib.pyplot as plt

#Q2.1.5
#Read the image and convert to grayscale, if necessary
I1 = cv2.imread('../data/cv_cover.jpg')
# img = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
# hist = []
rot = [10,20,30]
hist_sum = []
for i in range(36):
    print('%d th rot processing.....'%i)
	#Rotate Image
    rot_img = rotate(I1, i*10, reshape=True)
	#Compute features, descriptors and Match features
    matches, locs, rot_locs = matchPics(I1, rot_img)
    # if i in rot:
        # plotMatches(I1, rot_img, matches, locs, rot_locs, '%d rot match.jpeg'%i)
	# Update histogram
    hist_, bin_ = np.histogram(matches[:,0])
    hist_sum.append(sum(hist_))
    # plt.figure()
    # if n ==0:
    #     hist = hist_
    #     bin = bin_
    # else:
    #     hist = np.vstack((hist, hist_))
    #     bin = np.vstack((bin, bin_))
    # n += 1


#Display histogram
# plt.plot(list(range(1,knn+1)), harris_eu.tolist())
plt.plot(list(range(0,360,10)), hist_sum)
# plt.ylim(0,1000)
plt.show()
# plt.savefig('hist_result.jpeg')
# plt.close()