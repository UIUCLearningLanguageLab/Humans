import numpy as np
import scipy.stats as ss
from Programs.Graphical_Models import graphical_analysis


def calculate_rank_matrix(matrix,version): # get the standard ranking based on linguistic corpus, and get the model
    # syntagmatic ranking from semantic relatedness output.
    transpose = matrix.transpose()
    (m,n)= matrix.shape
    rank_matrix = np.zeros((n,m))
    if version == 'standard':
        # create the standard ranking from corpus information
        for i in range(n):
            occur_list = list(transpose[i])
            occurred = []
            not_occurred = []
            for j in range(m):
                if occur_list[j] > 0:
                    occurred.append(j)
                else:
                    not_occurred.append(j)
            if len(not_occurred) > 0:
                for j in not_occurred:
                    sim = 0
                    v1 = matrix[j]
                    for k in occurred:
                        v2 = matrix[k]
                        sim += np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    sim = sim/len(occurred)
                    transpose[i][j] = sim - 1
    for i in range(n):
        occur_list = list(transpose[i])
        rank_matrix[i] = ss.rankdata(occur_list)

    flat_ranking = rank_matrix.flatten()

    return flat_ranking


def get_task_matrix(kit, model):
    p_nouns = kit['p_nouns']
    t_verbs = kit['t_verbs']
    pairs = kit['pairs']
    vocab_index_dict = kit['vocab_index_dict']
    vocab_list = kit['vocab_list']
    task_matrix = np.zeros((len(p_nouns),len(t_verbs)))
    if model == 'cooc' or model == 'cooc_graph':
        grand_matrix = kit['cooc_matrix']
    else:
        grand_matrix = kit['sim_matrix']


    if model == 'cooc' or model == 'sim':
        for phrase in pairs:
            id_noun = p_nouns.index(phrase[1])
            id_verb = t_verbs.index(phrase[0])
            id_noun_grand = vocab_index_dict[phrase[1]]
            id_verb_grand = vocab_index_dict[phrase[0]]
            task_matrix[id_noun][id_verb] = grand_matrix[id_noun_grand][id_verb_grand]
    else:
        task_matrix = graphical_analysis.get_sr_matrix(grand_matrix, p_nouns, t_verbs, vocab_list)

    return task_matrix


def get_standard_ranking(kit):
    p_nouns = kit['p_nouns']
    t_verbs = kit['t_verbs']
    pairs = kit['pairs']
    ranking = np.zeros((len(p_nouns),len(t_verbs)))
    for phrase in pairs:
        id_argument = p_nouns.index(phrase[1])
        id_predicate = t_verbs.index(phrase[0])
        ranking[id_argument][id_predicate] = pairs[phrase]
    standard_ranking = calculate_rank_matrix(ranking,'standard')

    return standard_ranking


def run_task(kit, model):
    # get model ranking from the sr_matrix carried out by the model
    # and then compute the model ranking corr to standard corr
    standard_ranking = get_standard_ranking(kit)
    sr_matrix = get_task_matrix(kit, model)
    model_ranking = calculate_rank_matrix(sr_matrix, 'non')
    model_corr = np.corrcoef(model_ranking, standard_ranking)[0][1]
    output_ranking = model_ranking.reshape(len(model_ranking),1)
    return model_corr, output_ranking





