import numpy as np
import cv2
import scipy.io as sio
from matplotlib import pyplot as plt
import skimage.feature

PATCHWIDTH = 9

def briefMatch(desc1,desc2,ratio=0.7):

	matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True)
	return matches



def plotMatches(im1,im2,matches,locs1,locs2):
	fig, ax = plt.subplots(nrows=1, ncols=1)
	im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
	im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
	plt.axis('off')
	skimage.feature.plot_matches(ax,im1,im2,locs1,locs2,matches,matches_color='r',only_matches=True)
	plt.show()

	return


def makeTestPattern(patchWidth, nbits):
	np.random.seed(0)
	compareX = patchWidth*patchWidth * np.random.random((nbits,1))
	compareX = np.floor(compareX).astype(int)
	np.random.seed(1)
	compareY = patchWidth*patchWidth * np.random.random((nbits,1))
	compareY = np.floor(compareY).astype(int)

	return (compareX, compareY)




def computePixel(img, idx1, idx2, width, center):
	halfWidth = width // 2
	col1 = idx1 % width - halfWidth
	row1 = idx1 // width - halfWidth
	col2 = idx2 % width - halfWidth
	row2 = idx2 // width - halfWidth
	return 1 if img[int(center[0]+row1)][int(center[1]+col1)] < img[int(center[0]+row2)][int(center[1]+col2)] else 0


def computeBrief(img, locs):
	patchWidth = 9
	nbits = 256
	compareX, compareY = makeTestPattern(patchWidth,nbits)
	m, n = img.shape

	halfWidth = patchWidth//2

	locs = np.array(list(filter(lambda x: halfWidth <= x[0] < m-halfWidth and halfWidth <= x[1] < n-halfWidth, locs)))
	desc = np.array([list(map(lambda x: computePixel(img, x[0], x[1], patchWidth, c), zip(compareX, compareY))) for c in locs])

	return desc, locs


def corner_detection(im, sigma=1.5):
	# fast method
	result_img = skimage.feature.corner_fast(im, PATCHWIDTH)
	locs = skimage.feature.corner_peaks(result_img, min_distance=1)
	return locs
