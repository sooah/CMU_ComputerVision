import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

def matchPics(I1, I2):
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	img1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	# fast = cv2.FastFeatureDetector_create(threshold = 100, nonmaxSuppression=True)
	locs1 = corner_detection(img1)
	locs2 = corner_detection(img2)

	#Obtain descriptors for the computed feature locations
    #
	img1_decs, img1_locs = computeBrief(img1, locs1)
	img2_decs, img2_locs = computeBrief(img2, locs2)

	#Match features using the descriptors
	matches = briefMatch(img1_decs, img2_decs)
	locs1[:,:] = locs1[:,[1,0]]
	locs2[:,:] = locs2[:,[1,0]]

	return matches, locs1, locs2