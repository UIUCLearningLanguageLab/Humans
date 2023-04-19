import numpy as np
import scipy.stats as ss
import csv
import math
import matplotlib.pyplot as plt
from Programs.Graphical_Models import graphical_analysis
from pathlib import Path

########################################################################################################################
# Forming the verb-noun syntagmatic rankings of the models and the corpus gold standard
########################################################################################################################

plot_scatter = False
save_path = str(Path().cwd().parent / 'Data' /'rank_difference')
corr_flat = False

def get_ppmi_matrix(ww_matrix):  # get ppmi martix from co-occurrence matrix
    size = ww_matrix.shape
    ppmi_matrix = np.zeros(size)
    pmi_matrix = np.zeros(size)
    row_sum = ww_matrix.sum(1)
    column_sum = ww_matrix.sum(0)
    grand_sum = ww_matrix.sum()
    for i in range(size[0]):
        for j in range(size[1]):
            if ww_matrix[i][j] != 0:
                ppmi_matrix[i][j]=math.log2(ww_matrix[i][j]*grand_sum/(row_sum[i]*column_sum[j]))
                pmi_matrix[i][j] = ppmi_matrix[i][j]
                if ppmi_matrix[i][j] < 0:
                    ppmi_matrix[i][j] = 0
            else:
                ppmi_matrix[i][j] = 0
                pmi_matrix[i][j] = 0

    return ppmi_matrix, pmi_matrix

def get_log_row(ww_matrix):  # get the matrix row_logged
    size = ww_matrix.shape
    log_matrix = np.zeros(size)
    normalized_matrix = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            log_matrix[i][j]= math.log10(ww_matrix[i][j]+1)
    row_sum = log_matrix.sum(1)
    for i in range(size[0]):
        for j in range(size[1]):
            if row_sum[i] == 0:
                normalized_matrix[i][j]= 0
            else:
                normalized_matrix[i][j] = log_matrix[i][j]/row_sum[i]

    return normalized_matrix


def get_indirect_standard(to_eval, transpose, verb_id, not_occurred, occurred, measure, normalization):

    if normalization == 'ppmi': # use ppmi for normalization
        to_eval = get_ppmi_matrix(to_eval)[0]
    elif normalization == 'log': # use row_log to normalize
        to_eval = get_log_row(to_eval)

    if len(not_occurred) > 0:
        for j in not_occurred:

            sim = 0
            if len(occurred) > 0:
                v1 = to_eval[j]

                for k in occurred:
                    v2 = to_eval[k]

                    if measure == 'cos':  # when use cosine measure

                        n1 = np.linalg.norm(v1)
                        n2 = np.linalg.norm(v2)
                        if n1 != 0 and n2 != 0:
                            sim += np.inner(v1, v2) / (n1 * n2)
                        elif n1 == 0 and n2 == 0:
                            sim += 1


                    else:  # when use 2-distance
                        diff = v1 - v2
                        norm_1 = np.linalg.norm(diff)

                        if norm_1 != 0:
                            sim_2 = - norm_1
                            sim = sim + sim_2
                        else:
                            sim += 1
                sim = sim / len(occurred)

                transpose[verb_id][j] = sim - 1


    return transpose

def calculate_rank_matrix(matrix, version, evaluation_thematic, num_agent, measure=None, normalization = None):
    # get the standard ranking based on linguistic corpus, and
    # get the model syntagmatic ranking from semantic relatedness output. When getting standard ranking, there can be
    # different measure for paradigmatic similarity, and the measured vector could be separated by thematic roles, or
    # combined.

    # when the version is standard, the imported matrix should be event co-occurrence or syntagmatic rule, otherwise,
    # the imported matrix should be model semantic relatedness
    (m,n)= matrix.shape
    rank_matrix = np.zeros((n,m))

    transpose = matrix.transpose().copy()
    direct_dict = {} # record the (index of) the verbs which is involves purely direct relations

    if version == 'standard':
        # create the standard ranking from corpus information
        for i in range(n): # for every verb
            occur_list = list(transpose[i])


            if evaluation_thematic:
                direct_dict[i] = []
                occurred_a = []
                occurred_p = []
                not_occurred_a = []
                not_occurred_p = []
                for j in range(m): # for every noun
                    if j < num_agent: # if it is an agent
                        if occur_list[j] > 0:
                            occurred_a.append(j)
                        else:
                            not_occurred_a.append(j)
                    else: # if it is a patient
                        if occur_list[j] > 0:
                            occurred_p.append(j)
                        else:
                            not_occurred_p.append(j)

                if len(not_occurred_a) == 0: # if all nouns have occurred, the verb-role relation is direct
                    direct_dict[i].append('a')
                if len(not_occurred_p) == 0:
                    direct_dict[i].append('p')


                transpose = get_indirect_standard(matrix, transpose, i, not_occurred_a, occurred_a, measure, normalization)
                transpose = get_indirect_standard(matrix, transpose, i, not_occurred_p, occurred_p, measure, normalization)

            else:
                occurred = []
                not_occurred = []
                for j in range(m): # for evry noun
                    if occur_list[j] > 0:
                        occurred.append(j)
                    else:
                        not_occurred.append(j)

                if len(not_occurred) == 0: # if all nouns have occurred, the verb-role relation is direct
                    direct_dict[i] = 1
                transpose = get_indirect_standard(matrix, transpose, i, not_occurred, occurred, measure, normalization)

    trivialities = []

    for i in range(n):
        occur_list = list(transpose[i])
        if evaluation_thematic:
            rank_matrix[i][:num_agent]= ss.rankdata(occur_list[:num_agent])
            triviality_a = trivial_ranking(rank_matrix[i][:num_agent])
            trivialities.append(triviality_a)

            rank_matrix[i][num_agent:]= ss.rankdata(occur_list[num_agent:])
            triviality_p = trivial_ranking(rank_matrix[i][num_agent:])
            trivialities.append(triviality_p)
        else:
            rank_matrix[i] = ss.rankdata(occur_list)
            triviality = trivial_ranking(rank_matrix[i])
            trivialities.append(triviality)

    if corr_flat:
        return_ranking = rank_matrix.flatten()
        return_relate = transpose.flatten()
    else:
        return_ranking = rank_matrix
        return_relate = transpose



    return return_ranking, return_relate, trivialities, direct_dict

# transform a semantic relatedness matrix into the thematic-role form
def thematic_role_transfer(sr_matrix, kit, allow_thematic):
    if allow_thematic:
        nouns = kit['nouns']
    else:
        nouns = kit['noun_stems']
    verbs = kit['verbs']
    noun_dict = kit['noun_dict']
    noun_stems = kit['noun_stems']
    if allow_thematic:
        thematic_matrix = np.zeros((len(nouns), len(verbs)))
    else:
        thematic_matrix = np.zeros((len(noun_stems), len(verbs)))
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


# get the correct semantic relatedness matrix by the model parameter
def get_task_matrix(kit, encode, rep, dg, encoding_thematic, g_distance):
    if encoding_thematic:
        nouns = kit['nouns']
        pairs = kit['pairs']
    else:
        nouns = kit['noun_stems']
        pairs = kit['collapsed_pairs']

    verbs = kit['verbs']
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
            task_matrix[id_noun][id_verb] = grand_matrix[id_verb_grand][id_noun_grand]
    else:
        task_matrix = graphical_analysis.get_sr_matrix(grand_matrix, nouns, verbs, vocab_list, dg, g_distance)

    return task_matrix

def get_standard_ranking(kit, measure, evaluation_thematic, normalization):
    if evaluation_thematic:
        nouns = kit['nouns']
        pairs = kit['pairs']
    else:
        nouns = kit['noun_stems']
        pairs = kit['collapsed_pairs']
    verbs = kit['verbs']
    agent = kit['agent']
    num_agent = len(agent)

    rules = kit['rules'] # syntagmatic rule for city-block measure
    noun_stems = kit['noun_stems']
    noun_tax = kit['noun_tax']
    verb_indices = kit['verb_indices']
    ranking = np.zeros((len(nouns), len(verbs)))
    if measure == 'cos' or measure == 'dist': # use occurrence as standard and thus cosine or 2-distance for measure
        for phrase in pairs:
            id_argument = nouns.index(phrase[1])
            id_predicate = verbs.index(phrase[0])
            ranking[id_argument][id_predicate] = pairs[phrase]
        thematic_matrix = ranking


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

    standard_ranking, standard_thematic, standard_trivialities, direct_dict = calculate_rank_matrix(thematic_matrix,
                                                                                       'standard',evaluation_thematic,
                                                                                        num_agent, measure,
                                                                                       normalization)
    #print(thematic_matrix)
    #print()



    return standard_ranking, standard_thematic, standard_trivialities, direct_dict


def trivial_ranking(ranking):
    # telling if the ranking is trivial(all ranks are the same)
    flat_ranking = ranking.flatten()
    length = np.shape(flat_ranking)[0]
    triviality = True
    anchor = flat_ranking[0]
    for i in range(length):
        if flat_ranking[i] != anchor:
            triviality = False
            break
    return triviality


# calculate the rank difference (between the standard and the model prediction) by verb
def get_sorted_verb(model_re, corpus_rank, model_rank, verbs):
    flat_model_rank = model_rank.flatten()
    flat_corpus_rank = corpus_rank.flatten()
    rank_difference = abs(flat_model_rank - flat_corpus_rank)
    num_verb = len(verbs)
    num_noun = model_re.shape[0]
    verb_dict = {}

    for i in range(num_verb):
        verb_rank = rank_difference[i*num_noun: i*num_noun+num_noun]
        diff = verb_rank.sum()/num_noun
        verb_role = verbs[i]
        verb_dict[verb_role] = diff

    sorted_verb = sorted(verb_dict.items(), key=lambda x:x[1])
    return sorted_verb, flat_model_rank, flat_corpus_rank


# make the plot for rank difference

def plot_model_corpus(sorted_verb, model_num, flat_model_rank, flat_corpus_rank, model_corr):
    sorted_diff = []
    sorted_roles = []

    for item in sorted_verb:
        sorted_diff.append(item[1])
        sorted_roles.append(item[0])
    y_verb = np.arange(len(sorted_roles))

    fig, ax = plt.subplots(1,2,figsize=(9,9))
    fig.suptitle(model_num + ': ' + str(round(model_corr,3)))

    if int(model_num[1:]) % 2 == 0:
        color = 'blue'
    else:
        color = 'red'
    colors = []
    for i in y_verb:
        colors.append(color)
    ax[0].set_title('scattered ranks')
    ax[0].scatter(flat_corpus_rank, flat_model_rank)

    ax[1].set_title('rank difference by verbs')
    ax[1].barh(y_verb, sorted_diff, color = colors)
    ax[1].set_yticks(y_verb)
    ax[1].set_yticklabels(sorted_roles)
    ax[1].set_xlim(0,10)

    plt.tight_layout()
    #plt.show()


    plt.savefig(save_path + '/' + str(model_num) + '.png')



def run_task(kit, encode, rep, dg, evaluation_thematic, encoding_thematic, spearman, standards, verb_direct_dict,
             g_distance = None, ):
    # get model ranking from the sr_matrix carried out by the model
    # and then compute the model ranking corr to standard corr
    num_agent = len(kit['agent'])
    sr_matrix = get_task_matrix(kit, encode, rep, dg, encoding_thematic, g_distance)


    model_ranking, model_relate, model_trivialities, direct_dict = \
        calculate_rank_matrix(sr_matrix, 'non', evaluation_thematic, num_agent)

    if len(model_ranking.shape) == 2:
        (n_row, n_col) = model_ranking.shape
    else:
        n_row = 1
        n_col = model_ranking.shape[0]

    model_num = kit['model_num']
    verbs = kit['verbs']

    if evaluation_thematic:
        verb_corrs = np.zeros((2 * n_row,1))
    else:
        verb_corrs = np.zeros((n_row,1))

    standard_ranking = standards[0]
    standard_thematic = standards[1]
    standard_trivialities = standards[2]

    corr_sum = 0
    direct_corr = 0
    direct_num = 0
    indirect_corr = 0

    n_row_count = n_row

    if evaluation_thematic:

        if corr_flat:
            triviality = trivial_ranking(model_relate)
            if triviality:
                # print('trivial')
                model_corr = 0
            else:
                # print('not trivial')
                if spearman:
                    model_corr = ss.spearmanr(model_relate, standard_thematic)[0]
                else:
                    model_corr = np.corrcoef(model_ranking, standard_ranking)[0][1]
        else:
            for i in range(n_row):

                triviality_standard_a = standard_trivialities[2*i]
                triviality_standard_p = standard_trivialities[2*i+1]

                n_role = 2

                if triviality_standard_a:
                    verb_corr_a = 0
                    n_role = n_role - 1
                else:
                    # print('not trivial')
                    triviality_model_a = model_trivialities[2*i]
                    if triviality_model_a:
                        verb_corr_a = 0
                    else:
                        if spearman:
                            verb_corr_a = ss.spearmanr(model_relate[i][:num_agent],
                                                       standard_thematic[i][:num_agent])[0]
                            # print(verb_corr)
                        else:
                            verb_corr_a = np.corrcoef(model_ranking[i], standard_ranking[i])[0][1]

                if triviality_standard_p:
                    verb_corr_p = 0
                    n_role = n_role - 1
                else:
                    # print('not trivial')
                    triviality_model_p = model_trivialities[2*i+1]
                    if triviality_model_p:
                        verb_corr_p = 0
                    else:
                        if spearman:
                            verb_corr_p = ss.spearmanr(model_relate[i][num_agent:],
                                                       standard_thematic[i][num_agent:])[0]
                            # print(verb_corr)
                        else:
                            verb_corr_p = np.corrcoef(model_ranking[i], standard_ranking[i])[0][1]

                if n_role == 0:
                    n_row_count = n_row_count -1
                    verb_corrs[2*i] = 0
                    verb_corrs[2*i+1] = 0
                else:
                    if n_role == 1:
                        n_row_count = n_row_count - 0.5
                    verb_corrs[2*i] = verb_corr_a
                    verb_corrs[2*i+1] = verb_corr_p

                corr_sum = corr_sum + verb_corr_a + verb_corr_p

                directness = verb_direct_dict[i]
                if 'a' in directness:
                    direct_corr = direct_corr + verb_corr_a
                    direct_num = direct_num + 1
                else:
                    indirect_corr = indirect_corr + verb_corr_a
                if 'p' in directness:
                    direct_corr = direct_corr + verb_corr_p
                    direct_num = direct_num + 1
                else:
                    indirect_corr = indirect_corr + verb_corr_p

            model_corr = corr_sum / (2*n_row_count)
            direct_corr = direct_corr / direct_num
            indirect_corr = indirect_corr / (2*n_row_count-direct_num)
    else:
        if corr_flat:
            triviality = trivial_ranking(model_relate)
            if triviality:
                #print('trivial')
                model_corr = 0
            else:
                #print('not trivial')
                if spearman:
                    model_corr = ss.spearmanr(model_relate, standard_thematic)[0]
                else:
                    model_corr = np.corrcoef(model_ranking, standard_ranking)[0][1]
        else:
            for i in range(n_row):
                triviality_standard = standard_trivialities[i]

                if triviality_standard:
                    #print('trivial')
                    verb_corr = 0
                    n_row_count = n_row_count - 1
                else:
                    #print('not trivial')
                    triviality_model = model_trivialities[i]
                    if triviality_model:
                        verb_corr = 0
                    else:
                        if spearman:
                            verb_corr = ss.spearmanr(model_relate[i], standard_thematic[i])[0]
                            #print(verb_corr)
                        else:
                            verb_corr = np.corrcoef(model_ranking[i], standard_ranking[i])[0][1]

                corr_sum = corr_sum + verb_corr
                verb_corrs[i] = verb_corr

                if i in verb_direct_dict:
                    direct_num = direct_num + 1
                    direct_corr = direct_corr + verb_corr
                else:
                    indirect_corr = indirect_corr + verb_corr

            model_corr = corr_sum/n_row_count
            direct_corr = direct_corr/direct_num
            indirect_corr = indirect_corr/(n_row_count-direct_num)
    if plot_scatter:
        sorted_verb, flat_model_rank, flat_corpus_rank = get_sorted_verb(sr_matrix,standard_ranking,
                                                                        model_ranking,verbs)
        if model_corr > 0.76:
            plot_model_corpus(sorted_verb,model_num,flat_model_rank,flat_corpus_rank, model_corr)


    #for ranking in standard_rankings:
        #print(ranking)
        #print()


    output_ranking = model_ranking.reshape(n_row * n_col,1)
    output_relate = model_relate.reshape(n_row * n_col ,1)
    return model_corr, output_ranking, output_relate, verb_corrs, direct_corr, indirect_corr





