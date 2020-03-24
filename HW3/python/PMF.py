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

# train_mat = csr_matrix((rate_list, (user_list, movie_list)), shape = (user_len, movie_len))
# train_mat = csr_matrix((rate_list_pre, (user_list, movie_list)), shape = (user_len, movie_len))
rating_mat = train.pivot(index='user', columns='movie', values='rate')
# rating_mat = pd.DataFrame(train_mat.todense())
n_users, n_movies = rating_mat.shape
# n_users = user_len
index_list = rating_mat.index
columns_list = rating_mat.columns

min_rating, max_rating = train['rate'].min(), train['rate'].max()
rating_mat[rating_mat.isnull()] = -1
rating_mat = (rating_mat - min_rating)/(max_rating-min_rating)
# rating_mat[rating_mat==-0.5] = -1
rating_mat = torch.FloatTensor((rating_mat.values))

from torch.distributions import Normal, HalfNormal

rating_var = rating_mat.var()
class PMF(torch.nn.Module):
    def __init__(self, sigma_u = rating_var, sigma_v = rating_var):
        super().__init__()
        self.sigma = torch.tensor([1.], requires_grad=True)
        self.sigma_u = sigma_u
        self.sigma_v = sigma_v

    def forward(self, matrix, u_features, v_features):
        predicted = torch.sigmoid(torch.mm(u_features, v_features.t()))
        # predicted = torch.mm(u_features, v_features.t())
        likelihood = Normal(predicted, self.sigma)
        log_likelihood = likelihood.log_prob(matrix)

        non_zero_mask = (matrix != -1).type(torch.FloatTensor)
        total_llh = torch.sum(log_likelihood*non_zero_mask)

        u_prior = Normal(0, self.sigma_u)
        v_prior = Normal(0, self.sigma_v)
        logp_u = u_prior.log_prob(u_features).sum()
        logp_v = v_prior.log_prob(v_features).sum()

        logp_sig = HalfNormal(100.).log_prob(self.sigma).sum()

        result = total_llh + logp_u + logp_v + logp_sig

        return result

latentlist = [2, 5, 10, 20, 50]
for latent_vectors in latentlist:
    print(f'%d th latent vector'%latent_vectors)
# latent_vectors = 2
    user_features = torch.randn(n_users, latent_vectors, requires_grad=True)
    user_features.data.mul_(0.01)
    movie_features = torch.randn(n_movies, latent_vectors, requires_grad=True)
    movie_features.data.mul_(0.01)

    start_time = time.time()

    pmferror = PMF()
    optimizer = torch.optim.SGD([user_features, movie_features, pmferror.sigma], lr = 0.001, weight_decay=0.5)
    epochs = 1000
    for step in range(epochs):
        optimizer.zero_grad()
        loss = -pmferror(rating_mat, user_features, movie_features)
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

        if dev_user in index_list and dev_movie in columns_list:


        # def_df_user = train[train.user == dev_user]
        # user_list_tmp = def_df_user.user.tolist()
        # movie_list_tmp = def_df_user.movie.tolist()
        # #
        # # user_list_tmp = np.ones((user_len))
        # # movie_list_tmp = np.arange(movie_len)

            pred = torch.sigmoid(torch.mm(user_features[dev_user,:].view(1,-1), movie_features.t()))
            pred_rate = (pred*(max_rating-min_rating)+min_rating)
            pred_list = pred_rate.data.tolist()
            # result_mat_tmp = csr_matrix((pred_list, (user_list_tmp, movie_list_tmp)), shape=(1,movie_len))
            pred_result = pred_list[0][dev_movie]


            # pred = torch.mm(user_features[dev_user, :].view(1, -1), movie_features.t())+3
            # pred_list = pred.data.tolist()
            # # result_mat_tmp = csr_matrix((pred_list, (user_list_tmp, movie_list_tmp)), shape=(1,movie_len))
            # pred_result = pred_list[0][dev_movie]
        else:
            pred_result = 3.0


        file.writelines('%s\n'%(str(pred_result)))
        if i%10000==0:

            print('---------predicting for instance number %d' %i)
    file.close()
    print('%f secs spending'%(time.time()-start_time))