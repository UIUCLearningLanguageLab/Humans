import numpy as np
from Programs.Graphical_Models import STN
import networkx as nx
import math
VERBOSE = False


########################################################################################################################
# In sentHAL, co-occurrence count is carried out within sentence boundary, which is the co-occurrence of words within a
# sentence. Therefore, the only relevant model parameter is window weight, while window length varies by the size of
# sentence, and window type is ignored

# In synHAL, similar to sentHAL, count co-occurrence within a sentence. Instead of count the co-occurrence flat or
# linearly, the sentences are parsed into constituent trees and the co-occurrence is the inverse of the distance on the
# tree
########################################################################################################################

def create_tree_dict(corpus,linear_corpus):  # to encode the syntactical distance, first draw the constituent tree of
    # from the linear text
    tree_dict = {}
    vocab_list = []
    vocab_index_dict = {}
    for sentence in corpus:
        if sentence not in tree_dict:
            tree_edge = STN.complete_tree(sentence)[0]
            tree = nx.Graph()
            tree.add_edges_from(tree_edge)
            tree_dict[sentence] = tree
    for sentence in linear_corpus:
        for word in sentence:
            if word not in vocab_index_dict:
                vocab_index_dict[word] = len(vocab_list)
                vocab_list.append(word)
    return tree_dict, vocab_list, vocab_index_dict


def create_ww_matrix(tree_dict, vocab_list, vocab_index_dict, corpus, linear_corpus, window_weight):  # if window weight
    # is 'syntax', use the syntactic count, otherwise, use linear or flat
    num_vocab = len(vocab_list)
    count_matrix = np.zeros([num_vocab, num_vocab])
    for sentence in linear_corpus:
        parsed_sentence = corpus[linear_corpus.index(sentence)]
        tree = tree_dict[parsed_sentence]
        for i in range(len(sentence)-1):
            for j in range(len(sentence)-1-i):
                if sentence[i] in vocab_index_dict and sentence[j+1+i] in vocab_index_dict:
                    id1,id2 = vocab_list.index(sentence[i]), vocab_list.index(sentence[j+1+i])
                    if window_weight == 'syntax':
                        dist = 1/nx.shortest_path_length(tree,sentence[i],sentence[j+1+i])
                        count_matrix[id1,id2] += dist
                    elif window_weight == 'flat':
                        count_matrix[id1,id2] += 1
                    else:
                        count_matrix[id1,id2] += 1/(j+1)
    final_matrix = count_matrix + count_matrix.transpose()

    return final_matrix


def get_ppmi_matrix(ww_matrix):
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


def get_cos_sim(corpus,linear_corpus,source,target,window_weight,svd):
    tree_dict, vocab_list, vocab_index_dict = create_tree_dict(corpus, linear_corpus)
    final_matrix = create_ww_matrix(tree_dict, vocab_list, vocab_index_dict, corpus, linear_corpus, window_weight)
    ppmi_matrix, pmi_matrix = get_ppmi_matrix(final_matrix)
    if svd:
        ppmi_matrix = np.linalg.svd(ppmi_matrix)[0]
    if VERBOSE:
        print(vocab_list)
    # print(final_matrix)
    # print(pmi_matrix)
    cos_sim = {}
    v1 = ppmi_matrix[vocab_index_dict[source]]
    if VERBOSE:
        print(source)
        print(v1)
    for word in target:
        v2 = ppmi_matrix[vocab_index_dict[word]]
        if VERBOSE:
            print(word)
            print(v2)
        cos_sim[word]=np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos_sim

