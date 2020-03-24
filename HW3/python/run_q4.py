import torch
import pandas as pd
from scipy.sparse import csr_matrix
import time
from sklearn.metrics.pairwise import cosine_similarity

from utils import *
from module import *

train_path = 'data/train.csv'
train, movie_list, user_list, rate_list, rate_list_pre = bring_train(train_path, ['movie', 'user', 'rate', 'date'])
movie_len = max(movie_list)+1
user_len = max(user_list)+1

train_mat = csr_matrix((rate_list, (user_list, movie_list)), shape = (user_len, movie_len))
# train_mat = train_mat.T
# # trainPre_mat = csr_matrix((rate_list_pre, (movie_list, user_list)), shape=(movie_len, user_len))
# # trainPre_mat = trainPre_mat.T
# # rating_mat = torch.FloatTensor(rate_list)
#
# non_zero_ones = np.ones((len(rate_list)))
# non_zero_mat = csr_matrix((non_zero_ones, (movie_list, user_list)), shape = (movie_len, user_len))
rating_mat = train.pivot(index = 'user', columns='movie', values = 'rate')
# rating_mat = pd.DataFrame(train_mat.todense())
# min_rate, max_rate = train['rate'].min(), train['rate'].max()
# rating_mat = (rating_mat - min_rate)/(max_rate-min_rate)
n_users, n_movies = rating_mat.shape
rating_mat[rating_mat.isnull()] = -1
rating_mat = torch.FloatTensor(rating_mat.values)

latent_vectors = 2
user_features = torch.randn(n_users, latent_vectors, requires_grad = True)
user_features.data.mul_(0.01)
movie_features = torch.randn(n_movies, latent_vectors, requires_grad = True)
movie_features.data.mul_(0.01)

rating_var = rating_mat.var()
class PMF(torch.nn.Module):
    def __init__(self, u_lambda = 0.2, v_lambda = 0.2):
        super().__init__()
        self.u_lambda = u_lambda
        self.v_lambda = v_lambda
    def forward(self, mat, u_features, v_features):
        non_zero_mask = (mat != -1).type(torch.FloatTensor)
        pred = torch.mm(u_features, v_features.t())
        diff = (mat-pred)**2
        pred_err = torch.sum(diff*non_zero_mask)

        u_regularization = self.u_lambda*torch.sum(u_features.norm(dim=1))
        v_regularization = self.v_lambda*torch.sum(v_features.norm(dim=1))

        result = pred_err + u_regularization+v_regularization

        return result

start_time = time.time()
# loss_ = PMF()
# loss = loss_(rating_mat, user_features, movie_features)
optimizer = torch.optim.SGD([user_features, movie_features], lr = 0.01, weight_decay=0.5)
pmferr = PMF(u_lambda = rating_var, v_lambda = rating_var)
for step, epoch in enumerate(range(10)):
    optimizer.zero_grad()
    loss = pmferr(rating_mat, user_features, movie_features)
    loss.backward()
    optimizer.step()
    if step%50 == 0:
        print(f'Step {step}, {loss:.3f}')

dev_csv_path = 'data/dev.csv'
dev_df = pd.read_csv(dev_csv_path, names = ['movie', 'user'])

file = open('eval/PMF_%d.txt'%latent_vectors, 'w')
for i in range(len(dev_df.movie)):
    dev_movie = dev_df.iloc[i].movie
    dev_user = dev_df.iloc[i].user

    pred = torch.mm(user_features[dev_user,:].view(1,-1), movie_features.t())
    # pred_rate = (pred*(max_rate-min_rate)+min_rate)
    pred_result = pred[0,dev_movie].data.tolist()
    file.writelines('%s\n'%(str(pred_result)))
    print('---------predicting for instance number %d' %i)
file.close()
print('%f secs spending'%(time.time()-start_time))