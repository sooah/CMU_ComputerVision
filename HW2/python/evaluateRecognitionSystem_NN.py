import pickle
import numpy as np

from python.createFilterBank import create_filterbank
from python.getImageDistance import getImageDistance

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

K = 100
rand_eu_acc = []
rand_chi_acc = []
harris_eu_acc = []
harris_chi_acc = []

rand_eu_label = []
rand_chi_label = []
harris_eu_label = []
harris_chi_label = []

for i, path in enumerate(traintest['test_imagenames']):
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

    rand_dist_eu = getImageDistance(rand_hist, train_random_histset['trainFeatures'], 'euclidean')
    rand_eu_min_idx = np.argmin(rand_dist_eu)
    rand_eu_label.append(train_random_histset['trainLabels'][rand_eu_min_idx])

    rand_dist_chi = getImageDistance(rand_hist, train_random_histset['trainFeatures'], 'chi2')
    rand_chi_min_idx = np.argmin(rand_dist_chi)
    rand_chi_label.append(train_random_histset['trainLabels'][rand_chi_min_idx])

    harris_dist_eu = getImageDistance(harris_hist, train_harris_histset['trainFeatures'], 'euclidean')
    harris_eu_min_idx = np.argmin(harris_dist_eu)
    harris_eu_label.append(train_harris_histset['trainLabels'][harris_eu_min_idx])

    harris_dist_chi = getImageDistance(harris_hist, train_harris_histset['trainFeatures'], 'chi2')
    harris_chi_min_idx = np.argmin(harris_dist_chi)
    harris_chi_label.append(train_harris_histset['trainLabels'][harris_chi_min_idx])

labelresult = {'random_eu_label' : rand_eu_label, 'random_chi_label' : rand_chi_label, 'harris_eu_label' : harris_eu_label, 'harris_chi_label' : harris_chi_label}
with open('nn_labelresult.pkl', 'wb') as handle:
    pickle.dump(labelresult, handle, protocol = pickle.HIGHEST_PROTOCOL)

for n in range(len(traintest['test_imagenames'])):
    test_label = traintest['test_labels'][n]

    rand_eu_result = 1 if (test_label == rand_eu_label[n]) else 0
    rand_chi_result = 1 if (test_label == rand_chi_label[n]) else 0

    harris_eu_result = 1 if (test_label == harris_eu_label[n]) else 0
    harris_chi_result = 1 if (test_label == harris_chi_label[n]) else 0

    rand_eu_acc.append(rand_eu_result)
    rand_chi_acc.append(rand_chi_result)
    harris_eu_acc.append(harris_eu_result)
    harris_chi_acc.append(harris_chi_result)

acc_result = {'rand_eu_acc' : rand_eu_acc, 'rand_chi_acc' : rand_chi_acc, 'harris_eu_acc' : harris_eu_acc, 'harris_chi_acc' : harris_chi_acc}
with open('AccResult.pkl', 'wb') as handle:
    pickle.dump(acc_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
