# This module carries out analysis of Stns.
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import math
from scipy.sparse import lil_matrix

VERBOSE = False


########################################################################################################################
#  The following is a function for general analysis of a corpus and the STN generated from the corpus, which is more
#  meaningful when the corpus comes from naturalistic data. For artificially generated corpus, the information of the
#  corpus is part of the design, thus the analysis become less necessary
########################################################################################################################
def general_analysis(steven):

    # generate the lexical network out of the STN
    C_net = steven.constituent_net

    # get the nouns, verbs, adjs and advs and show their numbers.
    w_list = steven.word_list
    n_list = steven.tagged_word['n']
    v_list = steven.tagged_word['v']
    a_list = steven.tagged_word['a']
    r_list = steven.tagged_word['r']
    if VERBOSE:
        print()
        print(w_list)
        print('{} words in total'.format(len(w_list)))
        print('{} nouns in total'.format(len(n_list)))
        print('{} verbs in total'.format(len(v_list)))
        print('{} adjectives in total'.format(len(a_list)))
        print('{} adverbs in total'.format(len(r_list)))
        print()

    # print(diamonds)

    # primary information of the lexical net: num of words, and num of connections.
    if VERBOSE:
        print(C_net.number_of_nodes(), C_net.number_of_edges())

    # sizes of components in the lexical nets, and the distribution of sizes in histogram
        print('size of components')
    components = [len(c) for c in sorted(nx.connected_components(C_net), key=len, reverse=True)]
    if VERBOSE:
        print(components)
        print()
    degree = nx.degree_histogram(C_net)
    degree_distribution = {}
    for i in range(len(degree)):
        if degree[i] > 0:
            degree_distribution[i + 1] = degree[i]
    if VERBOSE:
        print(degree_distribution)
        print()

    # degree distribution of the nodes in
    degree_dict = {}

    for node in C_net:
        degree_dict[node] = C_net.degree(node)

    sorted_degree_dict = sorted(degree_dict.items(), key=operator.itemgetter(1), reverse=True)
    if VERBOSE:
        print('degree distribution of words in lexical net:')
        print(sorted_degree_dict)

    # to see if the distribution of the degrees follows the power law
    x = range(len(degree))
    y = [z / float(sum(degree)) for z in degree]
    plt.loglog(x, y, color='blue', linewidth=2)
    plt.show()

########################################################################################################################
#  activation spreading as a measure for semantic relatedness on STN and other graphical semantic models
#  it measures the functional distance instead of the structure distance on a network, by taking account of both the
#  structure distance and the relative strength of the connections in the network.

#  activation spreading is modeled by matrix multiplication: where a vector recording the current activation level of
#  each node get multiplied by the normalized weight matrix (of edges)
#  the semantic relatedness from node A to B, denoted be Re(A,B), is the amount of the activation that first transmitted
#  to B from A, where A is the only activated node initially, and during the spreading, at every moment, the nodes with
#  positive activation level sends out all its activation to its neighbors proportinal to the weights of the connections
#  to those neighbors
########################################################################################################################


def get_adjacency_matrix(net):
    # not necessary for linear models, which form the adjacency matrix directly
    # necessary for activation-spreading on complex network like STN
    W = nx.adjacency_matrix(net.network[2], nodelist=net.network[1])
    W.todense()
    length = W.shape[0]
    W = W + np.transpose(W)
    return W,length


def get_distance(adjacency_matrix, source, target, node_list):
    W = adjacency_matrix + np.transpose(adjacency_matrix)
    G = nx.from_numpy_matrix(W)
    source_index = node_list.index(source)
    semantic_relatedness_dict = {}
    for word in target:
        target_index = node_list.index(word)
        semantic_relatedness_dict[word] = 1/nx.shortest_path_length(G, source_index, target_index)
    return  semantic_relatedness_dict

def activation_spreading_analysis(adjacency_matrix, source, target, node_list, length):
    # Spreading activation to measure the functional distance from source to target
    # where source is one item, and target is a list(of items)
    # returns a sr_dictionary consisting of sr from the source to all targets


    activation = np.zeros((1,length),float)
    fired = np.ones((1, length), float)
    activation[0, node_list.index(source)] = 1
    fired[0, node_list.index(source)] = 0
    activation_recorder = activation

    while fired.any():
        activation = activation * adjacency_matrix
        activation_recorder = activation_recorder + np.multiply(fired, activation)
        for i in range(length):
            if fired[0,i] == 1 and activation[0,i] != 0:
                fired[0,i] = 0





    sorted_activation = activation_recorder.tolist()[0]
    sorted_activation.sort(reverse=True)
    node_dict = {}

    #  for node in node_list:
    #    node_dict[node] = activation_recorder[0,node_list.index(node)]
    #  sorted_dict = {k: v for k, v in sorted(node_dict.items(), key=lambda item: item[1], reverse=True)}
    #  for node in sorted_dict:
    #    print((node, sorted_dict[node]))

    semantic_relatedness_dict = {}
    for word in target:
        semantic_relatedness_dict[word] = activation_recorder[0, node_list.index(word)]


    return semantic_relatedness_dict


def get_activation_plot(net, activation_recorder, node_list):
    if net.network_type == 'stn':
        c_net = net.constituent_net
    else:
        c_net = net.network[2]

    color_list = []
    for node in c_net:
        color_list.append(math.log(activation_recorder[0, node_list.index(node)]))

    #  print(color_list)
    if VERBOSE:
        net.plot_lexical_network(color_list)
        plt.show()

def get_sr_matrix(adjacency_matrix, word_list1, word_list2, node_list, dg, g_distance):
    # word_list1 is for rows, and word_list2 for colums
    # in syntagmatic task, word_list1 are nouns, and word_list2 verbs
    # nouns are targets, while verbs are sources
    # return a matrix of semantic relatedness from sr_dictionaries
    sr_matrix = np.zeros((len(word_list1),len(word_list2)))

    #print(adjacency_matrix)
    if dg == False:
        W = adjacency_matrix + np.transpose(adjacency_matrix)
    else:
        W = adjacency_matrix

    # get rid of negative weights in graphs
    if W.min() < 0:
        W = (W + 1)/2

    W = np.asmatrix(W)
    length = W.shape[0]
    W = lil_matrix(W)


    if dg == False:
        normalizer = W.sum(1)
        for i in range(length):
            for j in range(length):
                if normalizer[i][0, 0] == 0:
                    W[i, j] = 0
                else:
                    W[i, j] = W[i, j] / normalizer[i][0, 0]

    for source in word_list2:
        if g_distance: # use graphical distance
            sr_dict = get_distance(W, source, word_list1, node_list)
        else:
            sr_dict = activation_spreading_analysis(W, source, word_list1, node_list, length)
        for target in word_list1:
            id1 = word_list1.index(target)
            id2 = word_list2.index(source)
            sr_matrix[id1][id2] = sr_dict[target]

    return sr_matrix
