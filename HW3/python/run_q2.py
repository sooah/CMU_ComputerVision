import pandas as pd
from scipy.sparse import csr_matrix

from utils import *
from module import *

from eval import eval_rmse

train_path = 'data/train.csv'
train, movie_list, user_list, rate_list, rate_list_pre = bring_train(train_path, ['movie', 'user', 'rate', 'date'])
movie_len = max(movie_list)+1
user_len = max(user_list)+1

# train_mat = csr_matrix((rate_list, (movie_list, user_list)), shape = (movie_len, user_len))
trainPre_mat = csr_matrix((rate_list_pre, (movie_list, user_list)), shape=(movie_len, user_len))

dev_csv_path = 'data/dev.csv'
# dev_query_path = 'data/dev.queries'

dev_df = pd.read_csv(dev_csv_path, names = ['movie', 'user'])
# dev_data, dev_index = load_query_data(dev_query_path, normalize=0)
# dev_data = list(map(lambda x:x-3, map(int, dev_data)))

# dev_data = list(map(int, dev_data))
# dev_q = csr_matrix((dev_data, dev_index), shape = (max(dev_index[0])+1, max(dev_index[1])+1)).T

# dev_uu_mat = (trainPre_mat.T)@dev_q
uu_mat = (trainPre_mat.T)@trainPre_mat
import time

# start_time = time.time()
# file = open('dev_pred_knn10_tie_heap.txt', 'w')
# for i in range(len(dev_df.movie)):
#     dev_df_movie = dev_df.iloc[i].movie
#     dev_df_user = dev_df.iloc[i].user
#
#     knn = 10
#     dot_knn_result = user_dot_knn_tie(uu_mat, dev_df_user, knn)
#     pred_list = []
#     try:
#         for j in range(knn):
#             pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
#             pred_list.append(pred_)
#         pred = sum(pred_list)/len(pred_list)
#     except:
#         pred = 3
#     file.writelines('%s\n'%(str(pred)))
#     print('-----predicting knn 10 for instance number %d'%i)
# file.close()
# print('%f secs spending'%(time.time()-start_time))

start_time = time.time()
file_knn100 = open('eval/dev_pred_knn100_tie_heap.txt', 'w')
for i in range(len(dev_df.movie)):
    dev_df_movie = dev_df.iloc[i].movie
    dev_df_user = dev_df.iloc[i].user

    knn = 100
    dot_knn_result = user_dot_knn_tie(uu_mat, dev_df_user, knn)
    pred_list = []
    try:
        for j in range(knn):
            pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
            pred_list.append(pred_)
        pred = sum(pred_list)/len(pred_list)
    except:
        pred = 3
    file_knn100.writelines('%s\n'%(str(pred)))
    print('-----predicting knn 100 for instance number %d'%i)
file_knn100.close()
print('%f secs spending'%(time.time()-start_time))

# #
start_time = time.time()
file_knn500 = open('eval/dev_pred_knn500_tie_heap.txt', 'w')
for i in range(len(dev_df.movie)):
    dev_df_movie = dev_df.iloc[i].movie
    dev_df_user = dev_df.iloc[i].user

    knn = 500
    dot_knn_result = user_dot_knn_tie(uu_mat, dev_df_user, knn)
    pred_list = []
    try:
        for j in range(knn):
            pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
            pred_list.append(pred_)
        pred = sum(pred_list)/len(pred_list)
    except:
        pred = 3
    file_knn500.writelines('%s\n'%(str(pred)))
    print('-----predicting knn 500 for instance number %d'%i)
file_knn500.close()
print('%f secs spending'%(time.time()-start_time))

from sklearn.metrics.pairwise import cosine_similarity
# #
uu_cos = cosine_similarity(trainPre_mat.T)
start_time = time.time()
file_cos_knn10 = open('cos_dev_pred_knn10_heap.txt', 'w')
for i in range(len(dev_df.movie)):
    dev_df_movie = dev_df.iloc[i].movie
    dev_df_user = dev_df.iloc[i].user

    knn = 10
    dot_knn_result = cos_knn(uu_cos, dev_df_user, knn)
    pred_list = []
    try:
        for j in range(knn):
            pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
            pred_list.append(pred_)
        pred = sum(pred_list)/len(pred_list)
    except:
        pred = 3
    file_cos_knn10.writelines('%s\n'%(str(pred)))
    print('-----predicting case of cos & knn 10 for instance number %d'%i)
file_cos_knn10.close()
print('%f secs spending'%(time.time()-start_time))
# #
start_time = time.time()
file_cos_knn100 = open('cos_dev_pred_knn100_heap.txt', 'w')
for i in range(len(dev_df.movie)):
    dev_df_movie = dev_df.iloc[i].movie
    dev_df_user = dev_df.iloc[i].user

    knn = 100
    dot_knn_result = cos_knn(uu_cos, dev_df_user, knn)
    pred_list = []
    try:
        for j in range(knn):
            pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
            pred_list.append(pred_)
        pred = sum(pred_list)/len(pred_list)
    except:
        pred = 3
    file_cos_knn100.writelines('%s\n'%(str(pred)))
    print('-----predicting case of cos & knn 100 for instance number %d'%i)
file_cos_knn100.close()
print('%f secs spending'%(time.time()-start_time))
# #
start_time = time.time()
file_cos_knn500 = open('cos_dev_pred_knn500_heap.txt', 'w')
for i in range(len(dev_df.movie)):
    dev_df_movie = dev_df.iloc[i].movie
    dev_df_user = dev_df.iloc[i].user

    knn = 500
    dot_knn_result = cos_knn(uu_cos, dev_df_user, knn)
    pred_list = []
    try:
        for j in range(knn):
            pred_ = trainPre_mat[dev_df_movie, dot_knn_result[j]]+3
            pred_list.append(pred_)
        pred = sum(pred_list)/len(pred_list)
    except:
        pred = 3
    file_cos_knn500.writelines('%s\n'%(str(pred)))
    print('-----predicting case of cos & knn 500 for instance number %d'%i)
file_cos_knn500.close()
print('%f secs spending'%(time.time()-start_time))



# #
# dev_df_movie = dev_df.iloc[0].movie
# dev_df_user = dev_df.iloc[0].user
#
# # dev_q_vec = dev_q[:,dev_df_user]
#
# # dot_train_q = (trainPre_mat.T)@(dev_q_vec)
# knn = 100
# dot_knn_result = user_dot_knn(uu_mat, dev_df_user, knn)
# pred_list = []
# try:
#     for i in range(knn):
#         pred_ = trainPre_mat[dev_df_movie, dot_knn_result[i]]+3
#         pred_list.append(pred_)
#     pred = sum(pred_list)/len(pred_list)
# except:
#     pred = 3
# print(1)
