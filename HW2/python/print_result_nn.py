import pickle
import math
import cv2 as cv
from python.getVisualWords import get_visual_words

from python.extractFilterResponses import extract_filter_responses
from python.getHarrisPoints import get_harris_points
from python.getRandomPoints import get_random_points
from python.getDictionary import get_dictionary
import os
import glob
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from python.getImageFeatures import get_image_features
import time
from PIL import Image
import numpy as np


# NN alpha = 50 k = 100
traintest_file = open('../data/traintest.pkl', 'rb')
traintest = pickle.load(traintest_file)
traintest_file.close()
test_label = traintest['test_labels']

img_path = '../python/nn_labelresult_200_500.pkl'
with open(img_path, 'rb') as handle:
    label = pickle.load(handle)

rand_eu_label = label['random_eu_label']
rand_chi_label = label['random_chi_label']
harris_eu_label = label['harris_eu_label']
harris_chi_label = label['harris_chi_label']

acc_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/nn_AccResult_200_500.pkl'
with open(acc_path, 'rb') as handle:
    acc = pickle.load(handle)

rand_eu_acc = acc['rand_eu_acc']
rand_chi_acc = acc['rand_chi_acc']
harris_eu_acc = acc['harris_eu_acc']
harris_chi_acc = acc['harris_chi_acc']

rand_eu_mat = np.zeros((8,8))
for i in range(len(rand_eu_label)):
    gt = int(test_label[i])
    predict = int(rand_eu_label[i])
    rand_eu_mat[gt-1, predict-1] = rand_eu_mat[gt-1, predict-1] +1
print('random euclidean metric')
print('confusion matrix : ')
print(rand_eu_mat)
print('random points & euclidean distance : %f'%(sum(rand_eu_acc)/len(rand_eu_acc)))

rand_chi_mat = np.zeros((8, 8))
for i in range(len(rand_chi_label)):
    gt = int(test_label[i])
    predict = int(rand_chi_label[i])
    rand_chi_mat[gt - 1, predict - 1] = rand_chi_mat[gt - 1, predict - 1] + 1
print('[random chi metric]')
print('confusion matrix : ')
print(rand_chi_mat)
print('random points & chi distance : %f'%(sum(rand_chi_acc)/len(rand_chi_acc)))

harris_eu_mat = np.zeros((8, 8))
for i in range(len(harris_eu_label)):
    gt = int(test_label[i])
    predict = int(harris_eu_label[i])
    harris_eu_mat[gt - 1, predict - 1] = harris_eu_mat[gt - 1, predict - 1] + 1
print('[harris euclidean metric]')
print('confusion matrix : ')
print(harris_eu_mat)
print('harris points & euclidean distance : %f'%(sum(harris_eu_acc)/len(harris_eu_acc)))

harris_chi_mat = np.zeros((8, 8))
for i in range(len(harris_chi_label)):
    gt = int(test_label[i])
    predict = int(harris_chi_label[i])
    harris_chi_mat[gt - 1, predict - 1] = harris_chi_mat[gt - 1, predict - 1] + 1
print('[harris chi metric]')
print('confusion matrix : ')
print(harris_chi_mat)
print('harris points & chi distance : %f' %(sum(harris_chi_acc)/len(harris_chi_acc)))
