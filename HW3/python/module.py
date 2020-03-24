import pandas as pd
import numpy as np

def bring_train(path, headerName):
    df = pd.read_csv(path, names = headerName)
    movie_list = df.movie.tolist()
    movie_list = list(map(int, movie_list))

    user_list = df.user.tolist()
    user_list = list(map(int, user_list))

    rate_list = df.rate.tolist()
    rate_list = list(map(int, rate_list))
    rate_list_pre = list(map(lambda x:x-3, map(int, rate_list)))

    return df, movie_list, user_list, rate_list, rate_list_pre

def user_dot_knn(mat, ID, Knn):
    ID_vector = mat[:,ID]
    value = ID_vector.data
    idx = ID_vector.indices
    argsort_list = np.argsort(value)[::-1]
    indice_list = abs(idx[argsort_list]-ID)
    val_idx = np.lexsort((indice_list, value))[::-1][1:Knn+1,]
    knn_list = idx[val_idx]

    return knn_list

def user_dot_knn_tie(mat, ID, Knn):
    ID_vector = mat[:,ID]
    value = ID_vector.data
    idx = ID_vector.indices
    val_idx = np.argsort(value, kind='heapsort')[::-1][1:Knn+1,]
    knn_list = idx[val_idx]
    return knn_list

def movie_dot_knn(mat, ID, Knn):
    ID_vector = mat[ID,:]
    value = ID_vector.data
    idx = ID_vector.indices
    argsort_list = np.argsort(value)[::-1]
    indice_list = abs(idx[argsort_list]-ID)
    val_idx = np.lexsort((indice_list, value))[::-1][1:Knn+1,]
    knn_list = idx[val_idx]

    return knn_list

def cos_knn(mat, ID, Knn):
    ID_vector = mat[:,ID]
    idx_list = np.abs(np.arange(ID_vector.shape[0])[::-1]-ID)
    knn_list_ = np.lexsort((idx_list, ID_vector))
    knn_list = np.flip(knn_list_)[1:Knn+1,]
    # knn_list = np.argsort(ID_vector, kind='heapsort')[::-1][1:Knn+1,]
    return knn_list, ID_vector

def uu_mean(similarity_mat, train_mat, dev, knn, method, file_name):
    file = open(file_name, 'w')
    for i in range(len(dev.movie)):
        dev_movie = dev.iloc[i].movie
        dev_user = dev.iloc[i].user

        knn = knn
        if method == 'dot':
            dot_knn_result = user_dot_knn(similarity_mat, dev_user, knn)
        elif method == 'cos':
            dot_knn_result, _ = cos_knn(similarity_mat, dev_user, knn)
        pred_list = []
        try:
            for j in range(knn):
                pred_ = train_mat[dev_movie, dot_knn_result[j]]+3
                pred_list.append(pred_)
            pred = sum(pred_list)/len(pred_list)
        except:
            pred = 3
        file.writelines('%s\n'%(str(pred)))
        # print('---------predicting knn %d for instance number %d'%(knn, i))
    file.close()
    return 0


def uu_weighted(similarity_mat, train_mat, dev, knn, file_name):
    file = open(file_name, 'w')
    for i in range(len(dev.movie)):
        dev_movie = dev.iloc[i].movie
        dev_user = dev.iloc[i].user

        knn = knn

        dot_knn_result, cos_weight = cos_knn(similarity_mat, dev_user, knn)
        cos_weight_list = abs(cos_weight[dot_knn_result])
        cos_weight = cos_weight/max(sum(cos_weight_list), 0.00001)
        # cos_weight = abs(cos_weight)
        pred_list = []
        try:
            for j in range(knn):
                pred_ = (train_mat[dev_movie, dot_knn_result[j]] + 3) * (cos_weight[dot_knn_result[j],])
                pred_list.append(pred_)
            # pred_list = abs(pred_list)
            # pred = sum(pred_list) / len(pred_list)
            pred = sum(pred_list)
        except:
            pred = 3.0
        file.writelines('%s\n' % (str(pred)))
        # print('---------predicting knn %d for instance number %d' % (knn, i))
    file.close()
    return 0

def mm_mean(similarity_mat, train_mat, dev, knn, method, file_name):
    file = open(file_name, 'w')
    for i in range(len(dev.movie)):
        dev_movie = dev.iloc[i].movie
        dev_user = dev.iloc[i].user

        knn = knn
        if method == 'dot':
            dot_knn_result = movie_dot_knn(similarity_mat, dev_movie, knn)
        elif method == 'cos':
            dot_knn_result,_ = cos_knn(similarity_mat, dev_movie, knn)
        pred_list = []
        try:
            for j in range(knn):
                pred_ = (train_mat[dot_knn_result[j],dev_user] + 3)
                pred_list.append(pred_)
            pred = sum(pred_list) / len(pred_list)
        except:
            pred = 3
        file.writelines('%s\n' % (str(pred)))
        # print('---------predicting knn %d for instance number %d' % (knn, i))
    file.close()
    return 0

def mm_weighted(similarity_mat, train_mat, dev, knn, file_name):
    file = open(file_name, 'w')
    for i in range(len(dev.movie)):
        dev_movie = dev.iloc[i].movie
        dev_user = dev.iloc[i].user

        knn = knn

        dot_knn_result, cos_weight = cos_knn(similarity_mat, dev_movie, knn)
        cos_weight_list = abs(cos_weight[dot_knn_result])
        cos_weight = cos_weight/max(sum(cos_weight_list), 0.00001)
        # cos_weight = abs(cos_weight)
        pred_list = []
        try:
            for j in range(knn):
                pred_ = (train_mat[dot_knn_result[j], dev_user] + 3) * (cos_weight[dot_knn_result[j],])
                pred_list.append(pred_)
            # pred_list = abs(pred_list)
            # pred = sum(pred_list) / len(pred_list)
            pred = sum(pred_list)
        except:
            pred = 3.0
        file.writelines('%s\n' % (str(pred)))
        # print('---------predicting knn %d for instance number %d' % (knn, i))
    file.close()
    return 0