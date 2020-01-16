import pickle
import math
import cv2 as cv
from python.getVisualWords import get_visual_words

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

rand_eu_label_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_random_eu_label_200_500.pkl'
with open(rand_eu_label_path, 'rb') as handle:
    rand_eu_label = pickle.load(handle)
rand_chi_label_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_random_chi_label_200_500.pkl'
with open(rand_chi_label_path, 'rb') as handle:
    rand_chi_label = pickle.load(handle)
harris_eu_label_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_harris_eu_label_200_500.pkl'
with open(harris_eu_label_path, 'rb') as handle:
    harris_eu_label = pickle.load(handle)
harris_chi_label_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_harris_chi_label_200_500.pkl'
with open(harris_eu_label_path, 'rb') as handle:
    harris_chi_label = pickle.load(handle)

rand_eu_acc_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_rand_eu_acc_200_500.pkl'
with open(rand_eu_acc_path, 'rb') as handle:
    rand_eu_acc = pickle.load(handle)
rand_chi_acc_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_rand_chi_acc_200_500.pkl'
with open(rand_chi_acc_path, 'rb') as handle:
    rand_chi_acc = pickle.load(handle)
harris_eu_acc_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_harris_eu_acc_200_500.pkl'
with open(harris_eu_acc_path, 'rb') as handle:
    harris_eu_acc = pickle.load(handle)
harris_chi_acc_path = 'C:/Users/soual/workspace/CMU_Computer_Vision/hw2/cv_hw2_python/knn_harris_chi_acc_200_500.pkl'
with open(harris_chi_acc_path, 'rb') as handle:
    harris_chi_acc = pickle.load(handle)

knn = 40

rand_eu = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = rand_eu_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    rand_eu[k-1,0] = acc
max_acc = max(rand_eu)
knn_best = np.argmax(rand_eu)+1
rand_eu_mat = np.zeros((8,8))
for i in range(len(rand_eu_label[knn_best-1])):
    gt = int(test_label[i])
    predict = int(rand_eu_label[knn_best-1, i])
    rand_eu_mat[gt-1, predict-1] = rand_eu_mat[gt-1, predict-1] +1
print('[random euclidean metric]')
print('confusion matrix : ')
print(rand_eu_mat)
print('%d th knn is best - acc : %f' %(knn_best, max_acc))
plt.plot(list(range(1,knn+1)) , rand_eu.tolist())
# plt.show()
plt.savefig('knn_rand_euclidean_200_500.png')
plt.close()

rand_chi = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = rand_chi_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    rand_chi[k-1,0] = acc
max_acc = max(rand_chi)
knn_best = np.argmax(rand_chi)+1
rand_chi_mat = np.zeros((8, 8))
for i in range(len(rand_chi_label[knn_best-1])):
    gt = int(test_label[i])
    predict = int(rand_chi_label[knn_best-1,i])
    rand_chi_mat[gt - 1, predict - 1] = rand_chi_mat[gt - 1, predict - 1] + 1
print('[random chi metric]')
print('confusion matrix : ')
print(rand_chi_mat)
print('%d th knn is best - acc : %f' %(knn_best, max_acc))
plt.plot(list(range(1,knn+1)) , rand_chi.tolist())
plt.savefig('knn_rand_chi_200_500.png')
plt.close()

harris_eu = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = harris_eu_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    harris_eu[k-1,0] = acc
max_acc = max(harris_eu)
knn_best = np.argmax(harris_eu)+1
harris_eu_mat = np.zeros((8, 8))
for i in range(len(harris_eu_label[knn_best-1])):
    gt = int(test_label[i])
    predict = int(harris_eu_label[knn_best-1,i])
    harris_eu_mat[gt - 1, predict - 1] = harris_eu_mat[gt - 1, predict - 1] + 1
print('[harris euclidean metric]')
print('confusion matrix : ')
print(harris_eu_mat)
print('%d th knn is best - acc : %f' %(knn_best, max_acc))
plt.plot(list(range(1,knn+1)) , harris_eu.tolist())
plt.savefig('knn_harris_eu_200_500.png')
plt.close()

harris_chi = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = harris_chi_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    harris_chi[k-1,0] = acc
max_acc = max(harris_chi)
knn_best = np.argmax(harris_chi)+1
harris_chi_mat = np.zeros((8, 8))
for i in range(len(harris_chi_label[knn_best-1])):
    gt = int(test_label[i])
    predict = int(harris_chi_label[knn_best-1,i])
    harris_chi_mat[gt - 1, predict - 1] = harris_chi_mat[gt - 1, predict - 1] + 1
print('[harris chi metric]')
print('confusion matrix : ')
print(harris_chi_mat)
print('%d th knn is best - acc : %f' %(knn_best, max_acc))
plt.plot(list(range(1,knn+1)) , harris_chi.tolist())
plt.savefig('knn_harris_chi_200_500.png')
plt.close()