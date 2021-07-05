import math
import numpy as np
import random as rd
from cytoolz import itertoolz

PAD = '*PAD*'
VERBOSE = False
period = True

########################################################################################################################
# building HAL-style co-occurrence matrix, the output will be a cooc matirx
# the cooc matrix has normalization status: no, ppmi, row probability
# reduction: svd or no svd
########################################################################################################################

def create_cooc_matrix(vocab_list, vocab_index_dict, tokens, encoding, boundary):  # no function call overhead - twice as fast
    window_type = encoding['window_type']
    window_size = encoding['window_size']
    window_weight = encoding['window_weight']
    # count
    num_vocab = len(vocab_list)
    count_matrix = np.zeros([num_vocab, num_vocab])

    if VERBOSE:
        print('\nCounting word-word co-occurrences in {}-word moving window'.format(window_size))

    if not boundary:
        for i in range(window_size):
            tokens.append(PAD)
        windows = itertoolz.sliding_window(window_size + 1, tokens)  # + 1 because window consists of t2s only
        for window in windows:
            # print(window)
            if window[0] in vocab_index_dict:
                for i in range(window_size):
                    if window[i+1] in vocab_index_dict:
                        dist = 1/(i+1)
                        if window_weight == "linear":
                            count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i+1]]] += dist
                        elif window_weight == "flat":
                            count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i+1]]] += 1
    else:
        for token in tokens:
            sent = token.copy()
            for i in range(window_size):
                sent.append(PAD)
            windows = itertoolz.sliding_window(window_size + 1, sent)  # + 1 because window consists of t2s only
            for window in windows:
                # print(window)
                if window[0] in vocab_index_dict:
                    for i in range(window_size):
                        if window[i + 1] in vocab_index_dict:
                            dist = 1 / (i + 1)
                            if window_weight == "linear":
                                count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i + 1]]] += dist
                            elif window_weight == "flat":
                                count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i + 1]]] += 1


    # window_type
    if window_type == 'forward':
        cooc_matrix = count_matrix
    elif window_type == 'backward':
        cooc_matrix = count_matrix.transpose()
    elif window_type == 'summed':
        cooc_matrix = count_matrix + count_matrix.transpose()
    elif window_type == 'concatenated':
        cooc_matrix = np.concatenate((count_matrix, count_matrix.transpose()), axis=1)
    else:
        raise AttributeError('Invalid arg to "window_type".')

    row_sum = count_matrix.sum(1)
    for i in range(num_vocab):
        if row_sum[i] == 0:
            for j in range(num_vocab):
                if i != j:
                    count_matrix[i, j] = rd.uniform(0.001, 0.002)
    #  print('Shape of normalized matrix={}'.format(final_matrix.shape))

    return cooc_matrix

def get_ppmi_matrix(ww_matrix):  # get ppmi martix from co-occurrence matrix
    size = ww_matrix.shape
    ppmi_matrix = np.zeros(size)
    pmi_matrix = np.zeros(size)
    row_sum = ww_matrix.sum(1)
    column_sum = ww_matrix.sum(0)
    grand_sum = ww_matrix.sum()
    for i in range(size[0]):
        for j in range(size[1]):
            if ww_matrix[i][j] != 0:
                ppmi_matrix[i][j]=math.log2(ww_matrix[i][j]*grand_sum/(row_sum[i]*column_sum[j]))
                pmi_matrix[i][j] = ppmi_matrix[i][j]
                if ppmi_matrix[i][j] < 0:
                    ppmi_matrix[i][j] = 0
            else:
                ppmi_matrix[i][j] = 0
                pmi_matrix[i][j] = 0

    return ppmi_matrix, pmi_matrix

def get_log_row(ww_matrix):  # get the matrix row_logged
    size = ww_matrix.shape
    log_matrix = np.zeros(size)
    normalized_matrix = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            log_matrix[i][j]= math.log10(ww_matrix[i][j]+1)
    row_sum = log_matrix.sum(1)
    for i in range(size[0]):
        for j in range(size[1]):
            if row_sum[i] == 0:
                normalized_matrix[i][j]= 0
            else:
                normalized_matrix[i][j] = log_matrix[i][j]/row_sum[i]

    return normalized_matrix


# get the cooc_matrix, due to the parameter setting

def get_cooc_matrix(vocab_list, vocab_index_dict, word_bag, encoding, normalization, reduction, boundary):
    cooc_matrix = create_cooc_matrix(vocab_list, vocab_index_dict, word_bag, encoding, boundary)
    if normalization == 'ppmi':
        cooc_matrix = get_ppmi_matrix(cooc_matrix)[0]
    elif normalization == 'log':
        cooc_matrix = get_log_row(cooc_matrix)
    if reduction == 'svd':
        cooc_matrix, cooc_var = np.linalg.svd(cooc_matrix)[:2]
        r_cooc_matrix = cooc_matrix[:,:3]
        return r_cooc_matrix, cooc_var
    else:
        return cooc_matrix

