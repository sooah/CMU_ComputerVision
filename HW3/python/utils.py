"""
utils.py
==========
I/O utils
"""
from argparse import Namespace
from collections import defaultdict

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def load_query_data(path, normalize=3):
    """load_query_data
    Load query data.
    Input has format UserID MovieId:rating ...

    :param path:
    :param normalize:
    returns data, (row, col)
    """

    data, row, col, = [], [], []
    with open(path) as infile:
        for line in infile:
            u_id, *ratings = line.split(' ')
            u_id = int(u_id)
            scores = []
            for rating in ratings:
                mov_id, score = rating.split(':')
                mov_id = int(mov_id)
                score = int(score) - normalize
                if score != 0:
                    scores.append(score)
                    row.append(u_id)
                    col.append(mov_id)
            data.extend(scores)
    return data, (row, col)


def load_raw_review_data(path):
    """load_raw_review_data

    :param path:
    """
    mov_ids, u_ids, scores = [], [], []
    with open(path) as infile:
        for line in infile:
            mov_id, u_id, score, _ = line.split(',')
            mov_id = int(mov_id)
            u_id = int(u_id)
            score = int(score)
            mov_ids.append(mov_id)
            u_ids.append(u_id)
            scores.append(score)
    return np.array(mov_ids), np.array(u_ids), np.array(scores)


def load_review_data_matrix(path,
                            normalize=3,
                            matrix_func=csr_matrix):
    """load_review_data_matrix

    :param path:
    :param normalize:
    :param cosine_norm:
    """
    # use csr for userxuser
    # use csc for itemxitem

    # load raw data
    data, rows_cols  = load_review_data(path, normalize=normalize)
    # nU x nM

    X = matrix_func((data_raw, rows_cols))

    user_set = set(rows_cols[0])
    mov_set = set(rows_cols[1])

    rows, cols = rows_cols

    return Namespace(
        user_set=user_set,
        mov_set=mov_set,
        data=data,
        X=X,
        rows=rows,
        cols=cols)


def load_review_data(path, normalize=3):
    """load_review_data
    Load movie review data.
    Input has format MovieID1,UserID11,rating_score_for_UserID11_to_MovieID1.

    :param path:
    returns data, (row, col)
    """

    data, row, col = [], [], []
    with open(path) as infile:
        for line in infile:
            mov_id, u_id, score, _ = line.split(',')

            mov_id = int(mov_id)
            u_id = int(u_id)
            score = int(score) - normalize
            data.append(score)
            row.append(u_id)
            col.append(mov_id)

    data = np.array(data, dtype=np.float)
    row = np.array(row)
    col = np.array(col)

    return data, (row, col)
