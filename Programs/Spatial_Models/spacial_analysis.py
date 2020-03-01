import numpy as np
from Programs.Spatial_Models import cooc_matrix
from cytoolz import itertoolz

PAD = '*PAD*'
VERBOSE = False
period = True


def get_cos_sim(matrix,source,target, vocab_index_dict):
    #print(vocab_index_dict)
    v1 = matrix[vocab_index_dict[source]]
    v2 = matrix[vocab_index_dict[target]]
    if VERBOSE:
        print(source, target)
        print(v1, v2)
    cos_sim = np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos_sim


def get_sr_matrix(matrix, word_list1, word_list2, vocab_index_dict):
    sr_matrix = np.zeros((len(word_list1), len(word_list2)))
    for source in word_list1:
        id1 = vocab_index_dict[source]
        for target in word_list2:
            id2 = vocab_index_dict[target]
            sr_matrix[id1][id2] = get_cos_sim(matrix, source, target, vocab_index_dict)
    return sr_matrix


def build_cooc_space(word_list1, word_list2):
    cooc_matrix = np.zeros((len(word_list1), len(word_list2)))
    model = 'c_s'
    return model, cooc_matrix