from Programs.World import world, config
from Programs.Linear_Models import sim_space_analysis, activation_spreading
from Programs.Syntactic_Models import STN, synHAL_analysis
from Programs.Experiment import paradigmatic_task as p_task
from Programs.Experiment import syntagmatic_task as s_task
from pathlib import Path
import numpy as np
import scipy.stats as ss
import csv
import matplotlib.pyplot as plt
import math

VERBOSE = False

def running_world():  # running the world and get the corpus
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(config.World.num_turn):
        the_world.next_turn()
    # the_display = display.Display(the_world)
    # the_display.root.mainloop()
    num_consumed_animal = config.World.num_animals - len(the_world.animal_list)
    if VERBOSE:
        print('{} animals consumed.'.format(num_consumed_animal))
        print(the_world.consumption)
        print('{} epochs passed'.format(the_world.epoch))
    return the_world

def generate_a_world():
    the_world = running_world()
    matrices = []
    for human in the_world.human_list:
        corpus = human.corpus
        num_sentence = len(corpus)
        linear_corpus = human.linear_corpus
        Steve = human.get_activated_words()[1]
        linear_Doug = STN.Dg(human.linear_corpus)
        p_nouns = human.p_noun
        t_verbs = human.t_verb
        rank_size = len(p_nouns) * len(t_verbs)
        #print(p_nouns)
        #print(t_verbs)
        target = p_nouns
        pairs = human.t_p_pairs
        ranking = np.zeros((len(p_nouns),len(t_verbs)))
        for phrase in pairs:
            id_argument = p_nouns.index(phrase[1])
            id_predicate = t_verbs.index(phrase[0])
            ranking[id_argument][id_predicate] = pairs[phrase]
        #print(ranking)
        standard_ranking = s_task.calculate_rank_matrix(ranking,'standard')
        flat_standard = standard_ranking.flatten().reshape(rank_size,1)
        flat_item = []
        for verb in t_verbs:
            for noun in p_nouns:
                phrase = verb + '_' + noun
                flat_item.append(phrase)
        #print('standard')
        #print(standard_ranking)
        #recording_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
        data_matrix = flat_standard
        category_sim = []
        within_between = []
        return category_sim, within_between