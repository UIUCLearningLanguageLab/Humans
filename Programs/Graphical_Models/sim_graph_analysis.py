import numpy as np


def get_sr_matrix(word_list1, word_list2):
    sr_matrix = np.zeros((len(word_list1),len(word_list2)))
    return sr_matrix


def build_sim_graph(word_list1, word_list2, sim_matrix):
    sim_graph = np.zeros((len(word_list1), len(word_list2)))
    model = 's_s'
    return model, sim_graph

