import numpy as np
import math
from Programs.Spatial_Models import cooc_matrix
from cytoolz import itertoolz

PAD = '*PAD*'
VERBOSE = False
period = True

def get_matrix_connected(matrix, coeff):
    n_row = np.shape(matrix)[0]
    n_col = np.shape(matrix)[1]
    min_num = np.max(matrix)
    for i in range(n_row):
        for j in range(n_col):
            num = matrix[i][j]
            if num != 0 and num < min_num:
                min_num = num
    for i in range(n_row):
        for j in range(n_col):
            if matrix[i][j] == 0:
                matrix[i][j] = min_num/coeff
    return matrix


def get_sim(matrix,source,target, vocab_index_dict, sim_type):
    #print(vocab_index_dict)
    #print(matrix)
    v1 = matrix[vocab_index_dict[source]]
    v2 = matrix[vocab_index_dict[target]]
    if VERBOSE:
        print(source, target)
        print(v1, v2)

    if sim_type == 'cos':
        sim = np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    elif sim_type == 'distance':
        v = v1 - v2
        sim = 1/(math.sqrt(np.inner(v,v))+1)
    elif sim_type == 'corr':
        sim = np.corrcoef(v1,v2)[0][1]
    #print(sim_type,sim)
    return sim


def get_sr_matrix(matrix, word_list1, word_list2, vocab_index_dict, sim_type):
    sr_matrix = np.zeros((len(word_list1), len(word_list2)))
    for source in word_list1:
        id1 = vocab_index_dict[source]
        for target in word_list2:
            id2 = vocab_index_dict[target]
            sr_matrix[id1][id2] = get_sim(matrix, source, target, vocab_index_dict, sim_type)
    connected_sr = get_matrix_connected(sr_matrix, 100)
    return connected_sr


def build_cooc_space(word_list1, word_list2):
    cooc_matrix = np.zeros((len(word_list1), len(word_list2)))
    model = 'c_s'
    return model, cooc_matrix