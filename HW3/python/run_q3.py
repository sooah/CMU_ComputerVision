import pandas as pd
from scipy.sparse import csr_matrix
import time
from sklearn.metrics.pairwise import cosine_similarity

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
dev_df = pd.read_csv(dev_csv_path, names = ['movie', 'user'])

uu_avg = trainPre_mat.mean(axis=0)

avg_ones = np.ones(trainPre_mat.data.shape)
ones_mat = csr_matrix((avg_ones, (movie_list, user_list)), shape=(movie_len, user_len))

avg_diag_row = list(range(user_len))
avg_diag = csr_matrix((uu_avg.A1, (avg_diag_row, avg_diag_row)), shape = (user_len, user_len))
uu_avg_mat = ones_mat@avg_diag
uu_cen = trainPre_mat - uu_avg_mat
uu_norm = np.linalg.norm(uu_cen.toarray(), 2)

uu_eq = np.divide(uu_cen, uu_norm)

pcc_dot = (uu_eq.T)@uu_eq
pcc_cos = cosine_similarity(uu_eq.T)

# print('[mean dot & knn = 10]')
# start_time = time.time()
# uu_mean(pcc_dot, trainPre_mat, dev_df, knn=10, method = 'dot', file_name = 'eval/pcc_mean_dot_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))

# print('[mean dot & knn = 100]')
# start_time = time.time()
# uu_mean(pcc_dot, trainPre_mat, dev_df, knn=100, method = 'dot', file_name = 'eval/pcc_mean_dot_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# print('[mean dot & knn = 500]')
# start_time = time.time()
# uu_mean(pcc_dot, trainPre_mat, dev_df, knn=500, method = 'dot', file_name = 'eval/pcc_mean_dot_knn500.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# print('[mean cos & knn = 10]')
# start_time = time.time()
# uu_mean(pcc_cos, trainPre_mat, dev_df, knn=10, method = 'cos', file_name = 'eval/pcc_mean_cos_knn10.txt')
# print('%f secs spending'%(time.time()-start_time))
#
# print('[mean cos & knn = 100]')
# start_time = time.time()
# uu_mean(pcc_cos, trainPre_mat, dev_df, knn=100, method = 'cos', file_name = 'eval/pcc_mean_cos_knn100.txt')
# print('%f secs spending'%(time.time()-start_time))

print('[mean cos & knn = 500]')
start_time = time.time()
uu_mean(pcc_cos, trainPre_mat, dev_df, knn=500, method = 'cos', file_name = 'eval/pcc_mean_cos_knn500.txt')
print('%f secs spending'%(time.time()-start_time))

print('[weighted cos & knn = 10]')
start_time = time.time()
uu_weighted(pcc_cos, trainPre_mat, dev_df, knn=10, file_name = 'eval/pcc_weighted_cos_knn10.txt')
print('%f secs spending'%(time.time()-start_time))

print('[weighted cos & knn = 100]')
start_time = time.time()
uu_weighted(pcc_cos, trainPre_mat, dev_df, knn=100, file_name = 'eval/pcc_weighted_cos_knn100.txt')
print('%f secs spending'%(time.time()-start_time))

print('[weighted cos & knn = 500]')
start_time = time.time()
uu_weighted(pcc_cos, trainPre_mat, dev_df, knn=500, file_name = 'eval/pcc_weighted_cos_knn500.txt')
print('%f secs spending'%(time.time()-start_time))