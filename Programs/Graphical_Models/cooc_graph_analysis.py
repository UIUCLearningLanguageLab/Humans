import networkx as nx
import matplotlib.pyplot as plt
import operator
import numpy as np
import math
from scipy.sparse import lil_matrix


def build_cooc_graph(word_list1, word_list2, cooc_matrix):
    cooc_graph = np.zeros((len(word_list1), len(word_list2)))
    model = 'c_g'
    return model, cooc_graph