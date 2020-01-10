# This module carries out analysis of Stns.
import STN
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import random
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


def activation_spreading_analysis(net, source, target):
    # Spreading activation to measure the functional distance

    W = nx.adjacency_matrix(net.network[2], nodelist = net.network[1])
    W.todense()
    l = W.shape[0]

    if net.network_type == 'stn':
        c_net = net.constituent_net
        W = W + np.transpose(W)
    else:
        c_net = net.network[2]
    W = lil_matrix(W)
    normalizer = W.sum(1)
    for i in range(l):
        for j in range(l):
            if normalizer[i][0,0] == 0:
                W[i,j] = 0
            else:
                W[i,j] = W[i,j]/normalizer[i][0,0]

    node_list = net.network[1]
    activation = np.zeros((1,l),float)
    fired = np.ones((1, l), float)
    activation[0, node_list.index(source)] = 1
    fired[0, node_list.index(source)] = 0
    activation_recorder = activation

    while fired.any():
        activation = activation*W
        activation_recorder = activation_recorder + np.multiply(fired, activation)
        for i in range(l):
            if fired[0,i] == 1 and activation[0,i] > 0:
                fired[0,i] = fired[0,i] - 1

    sorted_activation = activation_recorder.tolist()[0]
    sorted_activation.sort(reverse=True)
    node_dict = {}

    #  for node in node_list:
    #    node_dict[node] = activation_recorder[0,node_list.index(node)]
    #  sorted_dict = {k: v for k, v in sorted(node_dict.items(), key=lambda item: item[1], reverse=True)}
    #  for node in sorted_dict:
    #    print((node, sorted_dict[node]))

    color_list = []
    for node in c_net:
        color_list.append(math.log(activation_recorder[0, node_list.index(node)]))

    #  print(color_list)
    if VERBOSE:
        net.plot_lexical_network(color_list)
        plt.show()

    semantic_relatedness_dict = {}
    for word in target:
        semantic_relatedness_dict[word] = activation_recorder[0, node_list.index(word)]
    #  print(semantic_relatedness_dict)

    return semantic_relatedness_dict


