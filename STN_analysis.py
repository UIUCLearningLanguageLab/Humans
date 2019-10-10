# This module carries out analysis of Stns.
import STN
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import random
import math
from scipy.sparse import lil_matrix

def general_analysis(Steven):
    # get the corpus, which is a list of parsed sentences(in the form of embedded lists)

    # generate the STN corresponding to the network




    # generate the lexical network out of the STN
    C_net = Steven.constituent_net

    # generate some word lists to test the distance
    #word_list1 = ['milk', 'juice', 'tea', 'apple', 'peanut', 'potato', 'pear', 'tomato', 'carrot', 'noodle']
    #word_list2 = ['mommy', 'daddy', 'baby', 'doctor', 'teacher', 'farmer', 'girl', 'boy', 'grandma', 'mama']
    #word_list3 = ['dog', 'rabbit', 'squirrel', 'alligator', 'duck', 'goat', 'cat', 'chicken', 'horse', 'piggie']

    # get the nouns, verbs, adjs and advs and show their numbers.
    w_list = Steven.word_list
    n_list = Steven.tagged_word['n']
    v_list = Steven.tagged_word['v']
    a_list = Steven.tagged_word['a']
    r_list = Steven.tagged_word['r']

    print()
    print(w_list)
    print('{} words in total'.format(len(w_list)))
    print('{} nouns in total'.format(len(n_list)))
    print('{} verbs in total'.format(len(v_list)))
    print('{} adjectives in total'.format(len(a_list)))
    print('{} adverbs in total'.format(len(r_list)))
    print()

    # print(diamonds)

    # primrary information of the lexical net: num of words, and num of connections.
    print(C_net.number_of_nodes(), C_net.number_of_edges())

    # sizes of components in the lexical nets, and the distribution of sizes in histogram
    components = [len(c) for c in sorted(nx.connected_components(C_net), key=len, reverse=True)]
    print(components)
    print()
    degree = nx.degree_histogram(C_net)
    degree_distribution = {}
    for i in range(len(degree)):
        if degree[i] > 0:
            degree_distribution[i + 1] = degree[i]

    print(degree_distribution)
    print()

    # degree distribution of the nodes in
    degree_dict = {}

    for node in C_net:
        degree_dict[node] = C_net.degree(node)

    sorted_degree_dict = sorted(degree_dict.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_degree_dict)


    # to see if the distribution of the degrees follows the power law
    x = range(len(degree))
    y = [z / float(sum(degree)) for z in degree]
    plt.loglog(x, y, color='blue', linewidth=2)

    ##########################################################################

    # get the distances(relatedness) between the testing word list

    #time1 = time.time()

    #table1, matrix1 = Steven.compute_distance_matrix(word_list1, word_list2)
    #table2, matrix2 = Steven.compute_distance_matrix(word_list1, word_list3)
    #table3, matrix3 = Steven.compute_distance_matrix(word_list2, word_list3)

    #mean1 = matrix1.mean()
    #std1 = matrix1.std()
    #mean2 = matrix2.mean()
    #std2 = matrix2.std()
    #mean3 = matrix3.mean()
    #std3 = matrix3.std()
    # table2, matrix2= Steven.compute_similarity_matrix(w_list,1)

    #time2 = time.time()
    #print('Relatedness Matrix')
    #print(table1)
    #print(mean1, std1)
    #print()
    #print(table2)
    #print(mean2, std2)
    #print()
    #print(table3)
    #print(mean3, std3)
    #print()
    #time_used = time2 - time1
    #print('{} used for caculating the relatedness'.format(time_used))
    #print()


    # nx.draw(steven_adj, with_labels=True)
    # print('Similarity matrix')
    # print(table2)

    # nx.draw(C_net, with_labels=True)
    plt.show()


def activation_spreading_analysis(stn, words):
    # Spreading activation to measure the functional distance

    C_net = stn.constituent_net
    W = nx.adjacency_matrix(stn.network[2], nodelist = stn.network[1])
    W.todense()
    W = lil_matrix(W)
    l = W.shape[0]
    print(W)
    normalizer = W.sum(1)
    for i in range(l):
        for j in range(l):
            if normalizer[i][0,0] == 0:
                W[i,j] = 0
            else:
                W[i,j] = W[i,j]/normalizer[i][0,0]
    W = W + np.transpose(W)
    print(W)

    node_list = stn.network[1]
    activation = np.zeros((1,l),float)
    fired = np.ones((1, l), float)

    for word in words:
        activation[0, node_list.index(word)] = 1
        fired[0, node_list.index(word)] = 0
    activation_recorder = activation

    while fired.any():
        activation = activation*W
        activation_recorder = activation_recorder + np.multiply(fired, activation)
        for i in range(l):
            if fired[0,i] == 1 and activation[0,i] > 0:
                fired[0,i] = fired[0,i] - 1

    print(activation_recorder)
    sorted_activation = activation_recorder.tolist()[0]
    sorted_activation.sort(reverse = True)
    print(sorted_activation)

    color_list = []
    for node in C_net:
        scale = -math.log2(np.amin(activation_recorder))
        color = str(hex(round(255*(- math.log2(activation_recorder[0, node_list.index(node)]))/scale)))[2:]
        if color == '0':
            color_list.append('#ff0000')
        else:
            color_list.append('#ff'+color+color)

    print(color_list)

    stn.plot_network(color_list)
    plt.show()