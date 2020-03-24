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
# uu_mat = (trainPre_mat.T)@trainPre_mat
import time
from sklearn.metrics.pairwise import cosine_similarity

############ user-user ################
print('Start calculation for user-user...........')
uu_mat = (trainPre_mat.T)@trainPre_mat
uu_cos = cosine_similarity(trainPre_mat.T)
######### mean dot & knn = 10
# print('[mean dot & knn = 10]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=10, method='dot', file_name='eval/user_mean_dot_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean dot & knn = 100
# print('[mean dot & knn = 100]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=100, method='dot', file_name='eval/user_mean_dot_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean dot & knn = 500
# print('[mean dot & knn = 500]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=500, method='dot', file_name='eval/user_mean_dot_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean cos & knn = 10
# print('[mean cos & knn = 10]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=10, method='cos', file_name='eval/user_mean_cos_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean cos & knn = 100
# print('[mean cos & knn = 100]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=100, method='cos', file_name='eval/user_mean_cos_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean cos & knn = 500
# print('[mean cos & knn = 500]')
# start_time = time.time()
# uu_mean(uu_mat, trainPre_mat, dev_df, knn=500, method='cos', file_name='eval/user_mean_cos_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))

######### weighted knn=10
print('[weighted dot & knn = 10]')
start_time = time.time()
uu_weighted(uu_cos, trainPre_mat, dev_df, knn=10, file_name='eval/user_weighted_dot_knn10.txt')
print('%f secs spending'%(time.time()-start_time))

######### weighted knn=100
# start_time = time.time()
# uu_weighted(uu_cos, trainPre_mat, dev_df, knn=100, file_name='eval/weighted_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))


######### weighted knn=500
# start_time = time.time()
# file_cos_knn500 = open('eval/weighted_knn500.txt', 'w')
# uu_weighted(uu_cos, trainPre_mat, dev_df, knn=500, file_name='eval/weighted_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))

############ movie-movie ################
# print('Start calculation for movie-movie...........')
# mm_mat = trainPre_mat@(trainPre_mat.T)
# mm_cos = cosine_similarity(trainPre_mat)
#
# ######## mean dot & knn = 10
# print('[mean dot & knn = 10]')
# start_time = time.time()
# mm_mean(mm_mat, trainPre_mat, dev_df, knn=10, method='dot', file_name='eval/movie_mean_dot_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######## mean dot & knn = 100
# print('[mean dot & knn = 100]')
# start_time = time.time()
# mm_mean(mm_mat, trainPre_mat, dev_df, knn=100, method='dot', file_name='eval/movie_mean_dot_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######## mean dot & knn = 500
# print('[mean dot & knn = 500]')
# start_time = time.time()
# mm_mean(mm_mat, trainPre_mat, dev_df, knn=500, method='dot', file_name='eval/movie_mean_dot_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######## mean cos & knn = 10
# print('mean cos & knn = 10')
# start_time = time.time()
# mm_mean(mm_cos, trainPre_mat, dev_df, knn=10, method='cos', file_name='eval/movie_mean_cos_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean cos & knn = 100
# print('mean cos & knn = 100')
# start_time = time.time()
# mm_mean(mm_cos, trainPre_mat, dev_df, knn=100, method='cos', file_name='eval/movie_mean_cos_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# ######### mean cos & knn = 500
# print('mean cos & knn = 500')
# start_time = time.time()
# mm_mean(mm_cos, trainPre_mat, dev_df, knn=500, method='cos', file_name='eval/movie_mean_cos_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))