import numpy as np
import scipy.stats as ss
import csv
import matplotlib.pyplot as plt
from Programs.Graphical_Models import graphical_analysis
from pathlib import Path

########################################################################################################################
# Forming the verb-noun syntagmatic rankings of the models and the corpus gold standard
########################################################################################################################

plot_scatter = False
save_path = str(Path().cwd().parent / 'Data' /'rank_difference')

def calculate_rank_matrix(matrix,version, combine=None, measure=None):
    # get the standard ranking based on linguistic corpus, and
    # get the model syntagmatic ranking from semantic relatedness output. When getting standard ranking, there can be
    # different measure for paradigmatic similarity, and the measured vector could be separated by thematic roles, or
    # combined.

    # when the version is standard, the imported matrix should be event co-occurrence or syntagmatic rule, otherwise,
    # the imported matrix should be model semantic relatedness
    (m,n)= matrix.shape
    rank_matrix = np.zeros((n,m))

    transpose = matrix.transpose().copy()
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

            if combine == 'combine': # if combine, eveluate whole matrix
                evaluated = matrix
            else: # otherwise, split by roles
                num_verb = int(n/2)
                if i < num_verb:
                    evaluated = matrix[:,:num_verb]
                else:
                    evaluated = matrix[:,num_verb:]
            #print(evaluated)

            if len(not_occurred) > 0:
                for j in not_occurred:
                    sim = 0
                    if len(occurred) > 0:
                        v1 = evaluated[j]
                        for k in occurred:
                            v2 = evaluated[k]
                            if measure == 'cos':# when use cosine measure
                                n1 = np.linalg.norm(v1)
                                n2 = np.linalg.norm(v2)
                                if n1 != 0 and n2 != 0:
                                    sim += np.inner(v1, v2) / (n1*n2)
                                elif n1 == 0 and n2 == 0:
                                    sim += 1
                            else: # when use city-block measure
                                city_block = np.linalg.norm(v1-v2,1)
                                pos_agreement = np.inner(v1,v2)
                                neg_agreement = len(v1) - city_block - pos_agreement
                                #print(city_block, pos_agreement, neg_agreement)
                                if measure == 'city-block': # use city-block, consider both positive and negative agreements
                                    sim += len(v1)-city_block
                                elif measure == 'city-block_pos': # consider only positive agreement
                                    sim += pos_agreement
                                else: # consider only negative agreement
                                    sim += neg_agreement
                        sim = sim/len(occurred)
                    if measure == 'cos':
                        transpose[i][j] = sim - 1
                    else:
                        transpose[i][j] = sim
                for j in occurred:
                    if measure != 'cos':
                        transpose[i][j] = len(evaluated[j])

    for i in range(n):
        occur_list = list(transpose[i])
        rank_matrix[i] = ss.rankdata(occur_list)

    flat_ranking = rank_matrix.flatten()
    flat_relate = transpose.flatten()

    return flat_ranking, flat_relate

def thematic_role_transfer(sr_matrix, kit):
    nouns = kit['nouns']
    verbs = kit['verbs']
    noun_dict = kit['noun_dict']
    noun_stems = kit['noun_stems']
    thematic_matrix = np.zeros((len(noun_stems), 2 * len(verbs)))
    width = len(verbs)
    for noun in noun_dict:
        id_role_thematic = noun_stems.index(noun)
        roles = noun_dict[noun]
        for role in roles:
            id_role_sr = nouns.index(role)
            for verb in verbs:
                id_verb_sr = verbs.index(verb)
                if role[-1] == 'p': # when the noun has a patient role
                    id_verb_thematic = verbs.index(verb) + width
                else: # when the noun has a agent role
                    id_verb_thematic = verbs.index(verb)
                thematic_matrix[id_role_thematic][id_verb_thematic] = sr_matrix[id_role_sr][id_verb_sr]
    return thematic_matrix


def get_task_matrix(kit, encode, rep, dg):
    nouns = kit['nouns']
    verbs = kit['verbs']
    pairs = kit['pairs']
    vocab_index_dict = kit['vocab_index_dict']
    vocab_list = kit['vocab_list']
    task_matrix = np.zeros((len(nouns),len(verbs)))
    if encode == 'cooc':
        grand_matrix = kit['cooc_matrix']
    else:
        grand_matrix = kit['sim_matrix']

    if rep == 'space':
        for phrase in pairs:
            id_noun = nouns.index(phrase[1])
            id_verb = verbs.index(phrase[0])
            id_noun_grand = vocab_index_dict[phrase[1]]
            id_verb_grand = vocab_index_dict[phrase[0]]
            task_matrix[id_noun][id_verb] = grand_matrix[id_noun_grand][id_verb_grand]
    else:
        task_matrix = graphical_analysis.get_sr_matrix(grand_matrix, nouns, verbs, vocab_list, dg)
    #print(verbs)
    #print(nouns)
    #print(task_matrix)
    #print()
    return task_matrix

def get_standard_ranking(kit, combine, measure):
    nouns = kit['nouns']
    verbs = kit['verbs']
    pairs = kit['pairs']
    rules = kit['rules'] # syntagmatic rule for city-block measure
    noun_stems = kit['noun_stems']
    noun_tax = kit['noun_tax']
    verb_indices = kit['verb_indices']
    ranking = np.zeros((len(nouns), len(verbs)))
    if measure == 'cos': # use occurrence as standard and thus cosine for measure
        for phrase in pairs:
            id_argument = nouns.index(phrase[1])
            id_predicate = verbs.index(phrase[0])
            ranking[id_argument][id_predicate] = pairs[phrase]
        thematic_matrix = thematic_role_transfer(ranking,kit)
    else: # use syntagmatic rules as standard thus city-block as measure
        thematic_matrix = np.zeros((len(noun_stems), 2 * len(verbs)))
        for i in range(len(noun_stems)):
            noun = noun_stems[i]
            noun_category = noun_tax[noun]
            for verb in verbs:
                id_rule = verb_indices[verb]
                id_thematic = verbs.index(verb)
                noun_vec = rules[noun_category]
                thematic_matrix[i][id_thematic] = noun_vec[id_rule]
                thematic_matrix[i][id_thematic + len(verbs)] = rules[noun_category][id_rule + int(len(noun_vec)/2)]

    standard_ranking = calculate_rank_matrix(thematic_matrix,'standard',combine, measure)[0]
    #print(thematic_matrix)
    #print()
    return standard_ranking, thematic_matrix


def trivial_ranking(ranking):
    # telling if the ranking is trivial(all ranks are the same)
    length = np.shape(ranking)[0]
    triviality = True
    for i in range(length):
        if ranking[i] != ranking[0]:
            triviality = False
            break
    return triviality

def plot_model_corpus(corpus_re, model_re, corpus_rank, model_rank, model_num, verbs):
    flat_model_rank = model_rank.flatten()
    flat_corpus_rank = corpus_rank.flatten()
    rank_difference = abs(flat_model_rank - flat_corpus_rank)
    num_verb = len(verbs)
    num_noun = model_re.shape[0]
    verb_dict = {}
    for i in range(num_verb):
        verb_rank_a = rank_difference[i*num_noun: i*num_noun+num_noun]
        diff_a = verb_rank_a.sum()/num_noun
        verb_role = verbs[i] + '_a'
        verb_dict[verb_role] = diff_a
        verb_rank_p = rank_difference[(i + num_verb) * num_noun: (i + num_verb) * num_noun + num_noun]
        diff_p = verb_rank_p.sum()/num_noun
        verb_role = verbs[i] + '_p'
        verb_dict[verb_role] = diff_p





    sorted_verb = sorted(verb_dict.items(), key=lambda x:x[1])
    sorted_diff = []
    sorted_roles = []


    for item in sorted_verb[-10:]:
        sorted_diff.append(item[1])
        sorted_roles.append(item[0])
    y_verb = np.arange(len(sorted_roles))

    fig, ax = plt.subplots(1,2,figsize=(9,3))

    if model_num == 'M2':
        fig.suptitle('best graph')
    elif model_num == 'M7':
        fig.suptitle('best space')
    else:
        fig.suptitle(model_num)


    ax[0].set_title('scattered ranks')
    ax[0].scatter(flat_corpus_rank, flat_model_rank)

    ax[1].set_title('rank difference by verbs')
    ax[1].barh(y_verb, sorted_diff)
    ax[1].set_yticks(y_verb)
    ax[1].set_yticklabels(sorted_roles)
    ax[1].set_xlim(0,10)

    plt.tight_layout()

    if model_num == 'M2':
        plt.savefig(save_path + '/' + 'best_graph.png')
    elif model_num == 'M7':
        plt.savefig(save_path + '/' + 'best_space.png')

def run_task(kit, encode, rep, dg):
    # get model ranking from the sr_matrix carried out by the model
    # and then compute the model ranking corr to standard corr

    sr_matrix = get_task_matrix(kit, encode, rep, dg)
    thematic_matrix = thematic_role_transfer(sr_matrix, kit)
    model_ranking, model_relate = calculate_rank_matrix(thematic_matrix, 'non')
    model_num = kit['model_num']
    verbs = kit['verbs']
    if model_num == 'M2' or model_num == 'M7':
        print(model_ranking)
        print()
    #print(thematic_matrix)
    #print()

    measures = ['cos']
    combines = ['separate']
    standard_rankings = []
    model_corr_dict = {}
    triviality = trivial_ranking(model_ranking)

    for measure in measures:
        for combine in combines:
            #print(measure, combine)
            standard_ranking, standard_thematic = get_standard_ranking(kit,combine, measure)
            standard_rankings.append(standard_ranking)
            if model_num == 'M2':
                print('standard')
                print(standard_ranking)
                print()
            if triviality:
                model_corr = 0
            else:
                model_corr = np.corrcoef(model_ranking, standard_ranking)[0][1]
            if plot_scatter:
                plot_model_corpus(standard_thematic,thematic_matrix, standard_ranking, model_ranking,
                                  model_num, verbs)
            model_corr_dict[measure + '_' + combine] = model_corr


    #for ranking in standard_rankings:
    #    print(ranking)
    #    print()

    output_ranking = model_ranking.reshape(len(model_ranking),1)
    output_relate = model_relate.reshape(len(model_relate),1)
    return model_corr_dict, output_ranking, output_relate





