import pickle
import numpy as np
import time

from python.createFilterBank import create_filterbank
from python.getImageDistance import getImageDistance

import matplotlib.pyplot as plt

traintest_file = open('../data/traintest.pkl', 'rb')
traintest = pickle.load(traintest_file)
traintest_file.close()

test_imagenames = traintest['test_imagenames']
filterBank = create_filterbank()

with open('dictionaryRandom.pkl', 'rb') as handle:
    dict_random = pickle.load(handle)

with open('dictionaryHarris.pkl', 'rb') as handle:
    dict_harris = pickle.load(handle)

with open('visionRandom.pkl', 'rb') as handle:
    train_random_histset = pickle.load(handle)

with open('visionHarris.pkl', 'rb') as handle:
    train_harris_histset = pickle.load(handle)

K = 500
knn = 40

rand_eu_label = np.zeros((knn,len(traintest['test_imagenames'])))
rand_chi_label = np.zeros((knn,len(traintest['test_imagenames'])))
harris_eu_label = np.zeros((knn,len(traintest['test_imagenames'])))
harris_chi_label = np.zeros((knn,len(traintest['test_imagenames'])))

totalstarttime =time.time()
for i, path in enumerate(traintest['test_imagenames']):
    print('----processing %d image' % i)
    rand_wordmap_path = open('../data/%s_%s.pkl'%(path[:-4], 'Random'), 'rb')
    rand_wordmap = pickle.load(rand_wordmap_path)
    rand_wordmap_path.close()

    harris_wordmap_path = open('../data/%s_%s.pkl'%(path[:-4], 'Harris'), 'rb')
    harris_wordmap = pickle.load(harris_wordmap_path)
    harris_wordmap_path.close()

    rand_hist_ = np.histogram(rand_wordmap, bins = np.arange(K))
    rand_hist = rand_hist_[0]/np.sum(rand_hist_[0])
    rand_hist = rand_hist.reshape(-1, 1)
    harris_hist_ = np.histogram(harris_wordmap, bins = np.arange(K))
    harris_hist = harris_hist_[0]/np.sum(harris_hist_[0])
    harris_hist = harris_hist.reshape(-1, 1)

    starttime = time.time()
    for k in range(1,knn+1):
        rand_dist_eu = getImageDistance(rand_hist, train_random_histset['trainFeatures'], 'euclidean')
        rand_eu_min_list = np.argsort(rand_dist_eu)
        rand_eu_top_k = rand_eu_min_list[:k]
        rand_eu_top_labelset = (train_random_histset['trainLabels'][rand_eu_top_k]).astype(np.int64)
        rand_eu_k_label = np.bincount(rand_eu_top_labelset).argmax()
        rand_eu_label[k-1,i] = rand_eu_k_label

        rand_dist_chi = getImageDistance(rand_hist, train_random_histset['trainFeatures'], 'chi2')
        rand_chi_min_list = np.argsort(rand_dist_chi)
        rand_chi_top_k = rand_chi_min_list[:k]
        rand_chi_top_labelset = (train_random_histset['trainLabels'][rand_chi_top_k]).astype(np.int64)
        rand_chi_k_label = np.bincount(rand_chi_top_labelset).argmax()
        rand_chi_label[k-1,i] = rand_chi_k_label

        harris_dist_eu = getImageDistance(harris_hist, train_harris_histset['trainFeatures'], 'euclidean')
        harris_eu_min_list = np.argsort(harris_dist_eu)
        harris_eu_top_k = harris_eu_min_list[:k]
        harris_eu_top_labelset = (train_harris_histset['trainLabels'][harris_eu_top_k]).astype(np.int64)
        harris_eu_k_label = np.bincount(harris_eu_top_labelset).argmax()
        harris_eu_label[k-1,i] = harris_eu_k_label

        harris_dist_chi = getImageDistance(harris_hist, train_harris_histset['trainFeatures'], 'chi2')
        harris_chi_min_list = np.argsort(harris_dist_chi)
        harris_chi_top_k = harris_chi_min_list[:k]
        harris_chi_top_labelset = (train_harris_histset['trainLabels'][harris_chi_top_k]).astype(np.int64)
        harris_chi_k_label = np.bincount(harris_chi_top_labelset).argmax()
        harris_chi_label[k-1,i] = harris_chi_k_label

    print('For kNN ----- %f secs spending'%(time.time()-starttime))
print('get label %f sec'%(time.time()-totalstarttime))


with open('knn_random_eu_label.pkl', 'wb') as handle:
    pickle.dump(rand_eu_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('knn_random_chi_label.pkl', 'wb') as handle:
    pickle.dump(rand_chi_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('knn_harris_eu_label.pkl', 'wb') as handle:
    pickle.dump(harris_eu_label, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('knn_harris_chi_label.pkl', 'wb') as handle:
    pickle.dump(harris_chi_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

rand_eu_acc = np.zeros((knn,len(traintest['test_imagenames'])))
rand_chi_acc = np.zeros((knn,len(traintest['test_imagenames'])))
harris_eu_acc = np.zeros((knn,len(traintest['test_imagenames'])))
harris_chi_acc = np.zeros((knn,len(traintest['test_imagenames'])))

for n in range(len(traintest['test_imagenames'])):
    for k in range(1,knn+1):
        test_label = traintest['test_labels'][n]

        rand_eu_result = 1 if (test_label == rand_eu_label[k-1,n]) else 0
        rand_chi_result = 1 if (test_label == rand_chi_label[k-1,n]) else 0

        harris_eu_result = 1 if (test_label == harris_eu_label[k-1,n]) else 0
        harris_chi_result = 1 if (test_label == harris_chi_label[k-1,n]) else 0

        rand_eu_acc[k-1,n] = rand_eu_result
        rand_chi_acc[k-1,n] = rand_chi_result
        harris_eu_acc[k-1,n] = harris_eu_result
        harris_chi_acc[k-1,n] = harris_chi_result

with open('knn_rand_eu_acc.pkl', 'wb') as handle:
    pickle.dump(rand_eu_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('knn_rand_chi_acc.pkl', 'wb') as handle:
    pickle.dump(rand_chi_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('knn_harris_eu_acc.pkl', 'wb') as handle:
    pickle.dump(harris_eu_acc, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('knn_harris_chi_acc.pkl', 'wb') as handle:
    pickle.dump(harris_chi_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)

# check result on random & eu
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
plt.savefig('knn_rand_euclidean.png')
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
plt.savefig('knn_rand_chi.png')
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
plt.savefig('knn_harris_eu.png')
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
plt.savefig('knn_harris_chi.png')
plt.close()
