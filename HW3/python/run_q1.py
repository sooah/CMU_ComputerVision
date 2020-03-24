import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

from module import *

path = 'data/train.csv'

train = pd.read_csv(path, names= ['movie', 'user', 'rate', 'date'])
movie_list = train.movie.tolist()
movie_list = list(map(int, movie_list))
movie_len = max(movie_list)+1

user_list = train.user.tolist()
user_list = list(map(int, user_list))
user_len = max(user_list)+1

rate_list = train.rate.tolist()
rate_list = list(map(int, rate_list))
rate_list_pre = list(map(lambda x : x-3, map(int, rate_list)))

train_mat = csr_matrix((rate_list, (movie_list, user_list)), shape = (movie_len, user_len))
trainPre_mat = csr_matrix((rate_list, (movie_list, user_list)), shape=(movie_len, user_len))
# trainPre_mat = train_mat - 3

#1.1 statistics
numRate1 = train[(train.rate==1)]['movie'].nunique()
numRate3 = train[(train.rate==3)]['movie'].nunique()
numRate5 = train[(train.rate==5)]['movie'].nunique()
avg_rate = sum(rate_list)/len(rate_list)

#1.1 userID 4321
user4321Df = train[(train.user==4321)]
movie_4321_num = len(user4321Df['movie'])
numRate1_4321 = user4321Df[(user4321Df.rate==1)]['movie'].nunique()
numRate3_4321 = user4321Df[(user4321Df.rate==3)]['movie'].nunique()
numRate5_4321 = user4321Df[(user4321Df.rate==5)]['movie'].nunique()
avg_rate_4321 = sum(user4321Df.rate)/len(user4321Df.rate)

#1.1 movieID 3
movie3Df = train[(train.movie==3)]
user_movie3_num = len(movie3Df['user'])
numRate1_movie3 = movie3Df[(movie3Df.rate==1)]['user'].nunique()
numRate3_movie3 = movie3Df[(movie3Df.rate==3)]['user'].nunique()
numRate5_movie3 = movie3Df[(movie3Df.rate==5)]['user'].nunique()
avg_rate_movie3 = sum(movie3Df.rate)/len(movie3Df.rate)

#1.2 NN
# user4321 / dot
user4321 = trainPre_mat[:,4322]
dot_user4321 = (trainPre_mat.T)@user4321
uDotValue = dot_user4321.data
uDotidx = dot_user4321.indices
uDotknn5 = np.sort(uDotValue)[::-1][1:6,]
knn_userlist = []
for i in range(5):
    val = uDotknn5[i]
    idx = np.where(uDotValue == val)[0][0]
    knn_useridx = uDotidx[idx,]
    knn_userlist.append(knn_useridx)

# movie3 / dot
movie3 = trainPre_mat[3,:]
dot_movie3 = movie3@(trainPre_mat.T)
mDotValue = dot_movie3.data
mDotidx = (dot_movie3.T).indices
mDotknn5 = np.sort(mDotValue)[::-1][1:6,]
knn_movlist = []
for i in range(5):
    val = mDotknn5[i]
    idx = np.where(mDotValue == val)[0][0]
    knn_movidx = mDotidx[idx,]
    knn_movlist.append(knn_movidx)

# user4321 / cosine

from sklearn.preprocessing import normalize
user_norm = normalize(trainPre_mat, norm='l1', axis=0)

user4321_norm = user4321/np.sum(user4321.data)
dot_user_norm = (user_norm.T)@user_norm
uDotnormValue = dot_user_norm.data
uDotnormIdx = dot_user_norm.indices
uDotnormKnn5 = np.sort(uDotnormValue)[::-1][1:6,]
knn_userlist_norm = []
for i in range(5):
    val = uDotnormKnn5[i]
    idx = np.where(uDotnormValue == val)[0][0]
    knn_usernorm_idx = uDotnormIdx[idx,]
    knn_userlist_norm.append(knn_usernorm_idx)

# movie3 / cosine
movie_norm = normalize(trainPre_mat, norm='l1', axis=1)

movie3_norm = movie3/np.sum(movie3.data)
dot_movie3_norm = movie3_norm@(movie_norm.T)
mDotnormValue = dot_movie3_norm.data
mDotnormIdx = (dot_movie3_norm.T).indices
mDotnormKnn5 = np.sort(mDotnormValue)[::-1][1:6,]
knn_movlist_norm = []
for i in range(5):
    val = mDotnormKnn5[i]
    idx = np.where(mDotnormValue == val)[0][0]
    knn_movnorm_idx = mDotnormIdx[idx,]
    knn_movlist_norm.append(knn_movnorm_idx)

print(1)