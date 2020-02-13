from Programs.Linear_Models import sim_space_analysis, STN, graphical_analysis, synHAL_analysis
import numpy as np
import math


def get_category_sim(p_nouns, the_world, corpus, linear_corpus, encoding, model, svd): # get paradigmatic
    # relatedness(similarities) between noun pairs, also get mean similarities of noun categories and of within and
    # between category groups.
    num_category = 3
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
    noun_sim_matrix = np.zeros((len(p_nouns), len(p_nouns)-1))
    within_between = [np.array([0]),np.array([0])]
    for i in range(len(p_nouns)):
        noun = category_nouns[i]
        if model == 'hal':
            sim_noun = sim_space_analysis.get_cos_sim(linear_corpus, noun, category_nouns, encoding, svd)
        elif model == 'spatial':
            sim_noun = synHAL_analysis.get_cos_sim(corpus, linear_corpus, noun, category_nouns, window_weights, svd)
        else:
            sim_noun = graphical_analysis.activation_spreading_analysis(corpus, noun, category_nouns)
        for j in range(len(p_nouns)-1):
            target_noun = category_nouns[j]
            if target_noun != noun:
                noun_sim_matrix[i][j] = sim_noun[target_noun]
    #print(model)
    #print(noun_sim_matrix)

    category_sim_matrix = np.zeros((num_category, num_category))
    for i in range(num_category):
        r_s = category_nouns.index(categories[i][0])
        r_e = category_nouns.index(categories[i][-1])
        for j in range(num_category):
            c_s = category_nouns.index(categories[j][0])
            c_e = category_nouns.index(categories[j][-1])
            if i == j:
                sub_matrix = noun_sim_matrix[r_s:r_e + 1, c_s:c_e].flatten()
                within_between[0] = np.concatenate((within_between[0],sub_matrix),0)
            else:
                sub_matrix = noun_sim_matrix[r_s:r_e + 1, c_s:c_e + 1].flatten()
                within_between[1] = np.concatenate((within_between[1],sub_matrix),0)
            category_sim_matrix[i][j] = sub_matrix.mean()
    within_between[0] = within_between[0][1:]
    within_between[1] = within_between[1][1:]
    within_between = [(within_between[0].mean(),within_between[0].std()/math.sqrt(len(within_between[0]))),
                      (within_between[1].mean(),within_between[1].std()/math.sqrt(len(within_between[1])))]
    return category_sim_matrix, within_between