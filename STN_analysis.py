# This module carries out analysis of Stns.
import STN
import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import random
from scipy.sparse import lil_matrix

def analysis(corpus):
    # get the corpus, which is a list of parsed sentences(in the form of embedded lists)

    # generate the STN corresponding to the network
    Steven = STN.Stn(corpus)



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

    # Spreading activation to measure the functional distance
    W = nx.adjacency_matrix(Steven.network[2], nodelist = Steven.network[1])
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


    word = random.choice(w_list)
    activation = np.zeros((1,l),float)
    activation[0,Steven.network[1].index(word)] = 1
    for i in range(50):
        activation = activation*W
        for j in range(l):
            if activation[0,j] > 1:
                activation[0,j] = 1
        print(i)
        print(np.round(activation,2))





        








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

    Steven.plot_network()




    # nx.draw(C_net, with_labels=True)
    plt.show()