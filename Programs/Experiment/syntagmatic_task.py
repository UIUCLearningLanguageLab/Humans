import numpy as np
import scipy.stats as ss


def calculate_rank_matrix(matrix,version): # get the standard ranking based on linguistic corpus, and get the model
    # syntagmatic ranking from semantic relatedness output.
    transpose = matrix.transpose()
    (m,n)= matrix.shape
    rank_matrix = np.zeros((n,m))
    if version == 'standard':
        for i in range(n):
            occur_list = list(transpose[i])
            occurred = []
            not_occurred = []
            for j in range(m):
                if occur_list[j] > 0:
                    occurred.append(j)
                else:
                    not_occurred.append(j)
            if len(not_occurred) > 0:
                for j in not_occurred:
                    sim = 0
                    v1 = matrix[j]
                    for k in occurred:
                        v2 = matrix[k]
                        sim += np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    sim = sim/len(occurred)
                    transpose[i][j] = sim - 1
    for i in range(n):
        occur_list = list(transpose[i])
        rank_matrix[i] = ss.rankdata(occur_list)

    return rank_matrix


def get_model_ranking(sr_matrix):
    matrix_size = sr_matrix.shape()
    model_rank_matrix = np.zeros(matrix_size)
    return model_rank_matrix