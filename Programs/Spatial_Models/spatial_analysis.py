import numpy as np
import math
from Programs.Spatial_Models import cooc_matrix
from cytoolz import itertoolz

PAD = '*PAD*'
VERBOSE = False
period = True

########################################################################################################################
# spatial model analysis:
# compute semantic relatedness scores of the spatial models(similarity based), where scores are the similarities between
# word embeddings (co-occurrence row vectors)
########################################################################################################################

def trivial_ranking(ranking):
    # telling if the ranking is trivial(all ranks are the same)
    flat_ranking = ranking.flatten()
    length = np.shape(flat_ranking)[0]
    triviality = True
    for i in range(length):
        if flat_ranking[i] != flat_ranking[0]:
            triviality = False
            break
    return triviality


# need to make sure the matrix can be transformed to a connected graph (otherwise can not run activation spreading)
# basically, replace the zeroes in the matrix by a very small number, so, that the disconnected items are very weakly
# connected to other items
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


# compute the similarity score between two word vectors

def get_sim(matrix,source,target, vocab_index_dict, sim_type):
    #print(vocab_index_dict)
    #print(matrix)
    sim = 0
    v1 = matrix[vocab_index_dict[source]]
    v2 = matrix[vocab_index_dict[target]]
    if VERBOSE:
        print(source, target)
        print(v1, v2)
    if sim_type == 'cos' or sim_type == 'r_cos':
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 ==0 or norm2 ==0:
            if norm1 ==0 and norm1 ==0:
                sim = 1
            else:
                sim = 0
        else:
            sim = np.inner(v1,v2)/(norm2*norm1)
    elif sim_type == 'distance' or sim_type == 'r_distance':
        v = v1 - v2
        sim = 1/(math.sqrt(np.inner(v,v))+1)
    elif sim_type == 'corr' or sim_type == 'r_corr':
        t1 = trivial_ranking(v1)
        t2 = trivial_ranking(v2)
        if t1 != t2:
            sim = 0
        else:
            if t1:
                sim = 1
            else:
                sim = np.corrcoef(v1,v2)[0][1]
    #print(sim_type,sim)
    return sim


# get the semantic relatedness matrix

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