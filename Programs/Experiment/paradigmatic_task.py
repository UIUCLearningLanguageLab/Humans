from Programs.Graphical_Models import graphical_analysis
import numpy as np
import math


def get_categories(p_nouns, the_world):
    animals = []
    fruits = []
    drinks = []
    for noun in p_nouns:
        if noun in the_world.animal_category:
            animals.append(noun)
        elif noun in the_world.fruit_category:
            fruits.append(noun)
        else:
            drinks.append(noun)
    category_nouns = animals + fruits + drinks
    categories = [animals, fruits, drinks]
    return categories, category_nouns


def get_noun_sim(p_nouns, category_nouns, matrix, rep, vocab_list):
    noun_sim_matrix = np.zeros((len(p_nouns), len(p_nouns) - 1))
    for i in range(len(category_nouns)):
        noun = category_nouns[i]
        if rep == 'graph':
            grand_list = p_nouns
        else:
            grand_list = vocab_list
        id_noun = grand_list.index(noun)
        for j in range(len(p_nouns)-1):
            target_noun = category_nouns[j]
            if target_noun != noun:
                id_target = grand_list.index(target_noun)
                noun_sim_matrix[i][j] = matrix[id_noun][id_target]
    return noun_sim_matrix

def get_category_sim(category_nouns, categories, noun_sim_matrix): # get paradigmatic
    # relatedness(similarities) between noun pairs, also get mean similarities of noun categories and of within and
    # between category groups.
    num_category = len(categories)
    within_between = [np.array([0]),np.array([0])]
    category_sim_matrix = np.zeros((num_category, num_category))
    for i in range(num_category):
        r_s = category_nouns.index(categories[i][0])
        r_e = category_nouns.index(categories[i][-1])
        for j in range(num_category):
            c_s = category_nouns.index(categories[j][0])
            c_e = category_nouns.index(categories[j][-1])
            if i <= j:
                sub_matrix = noun_sim_matrix[r_s:r_e + 1, c_s:c_e].flatten()
                within_between[0] = np.concatenate((within_between[0],sub_matrix),0)
            else:
                sub_matrix = noun_sim_matrix[r_s:r_e + 1, c_s:c_e + 1].flatten()
                within_between[1] = np.concatenate((within_between[1],sub_matrix),0)
            category_sim_matrix[i][j] = sub_matrix.mean()
    within_between[0] = within_between[0][1:]
    within_between[1] = within_between[1][1:]
    within_between_result = [within_between[0].mean(),within_between[0].std()/math.sqrt(len(within_between[0])),
                      within_between[1].mean(),within_between[1].std()/math.sqrt(len(within_between[1]))]
    return category_sim_matrix, within_between_result


def run_task(kit, encode, rep):
    p_nouns = kit['p_nouns']
    the_world = kit['the_world']
    vocab_list = kit['vocab_list']
    if rep == 'space':
        matrix = kit['sim_matrix']

    elif encode == 'cooc':
        adjacency_m = kit['cooc_matrix']
        matrix = graphical_analysis.get_sr_matrix(adjacency_m, p_nouns, p_nouns, vocab_list)
    else:
        adjacency_m = kit['sim_matrix']
        matrix = graphical_analysis.get_sr_matrix(adjacency_m, p_nouns, p_nouns, vocab_list)
    categories, category_nouns = get_categories(p_nouns, the_world)
    noun_sim_matrix = get_noun_sim(p_nouns,category_nouns, matrix, rep, vocab_list)
    category_sim_matrix, within_between = get_category_sim(category_nouns, categories, noun_sim_matrix)
    return category_sim_matrix, within_between

