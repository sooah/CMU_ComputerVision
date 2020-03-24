import numpy as np
import cv2
import skimage.io 
import skimage.color

#Import necessary functions
from matchPics import matchPics
from helper import plotMatches
from planarH import *


#Write script for Q2.2.4
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

H, W, _ = cv_cover.shape
hp_cover_resized = cv2.resize(hp_cover, dsize=(W, H))

matches, locs1, locs2 = matchPics(cv_desk, hp_cover)
H2to1, inliers = computeH_ransac(locs1[matches[:,0]], locs2[matches[:,1]])
warped_img = cv2.warpPerspective(hp_cover_resized, H2to1, dsize=(cv_desk.shape[1], cv_desk.shape[0]))
cv2.imwrite('./result/HarryPotter_made.jpg', warped_img)
cv2.imshow('HarryPotter_made', warped_img)
cv2.waitKey(0)

composite_img = compositeH(H2to1, hp_cover_resized, cv_desk)
cv2.imwrite('./result/HarryPotter_Desk.jpg', composite_img)
cv2.imshow('HarryPotter_Desk', composite_img)
cv2.waitKey(0)