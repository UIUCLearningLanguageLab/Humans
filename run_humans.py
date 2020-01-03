from src import world
from src import config
import STN_analysis
import STN
import HAL_analysis
import synHAL_analysis
import numpy as np
import scipy.stats as ss
import random
from src.display import display
VERBOSE = False
window_types = ['forward','backward','summed']
window_sizes = [3,5,7,9]
window_weights = ['linear','flat']

stv = False
doug = False
hal = True
synhal = False
senthal = False


def check_word_in_list(the_list, the_dict):
    judge = 0
    for word in the_list:
        if word not in the_dict:
            judge = judge + 1
            break
    return judge


def running_world():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(config.World.num_turn):
        the_world.next_turn()
    #the_display = display.Display(the_world)
    #the_display.root.mainloop()
    num_consumed_animal = config.World.num_animals - len(the_world.animal_list)
    if VERBOSE:
        print('{} animals consumed.'.format(num_consumed_animal))
        print(the_world.consumption)
        print('{} epochs passed'.format(the_world.epoch))
    return the_world


def dict_to_rank(dict):
    list = []
    for word in dict:
        list.append(dict[word])
    rank = str(ss.rankdata(list))
    return rank


def one_ordering_task():
    the_world = running_world()
    matrices = []
    for human in the_world.human_list:
        #print('eat fruit {}'.format(human.eat_count_fruit))
        #print('eat meal {}'.format(human.eat_count_meal))
        #print('drink {}'.format(human.drink_count))
        #print('sleep {}'.format(human.sleep_count))
        #print('idle {}'.format(human.idle_count))
        corpus = human.corpus
        linear_corpus = human.linear_corpus
        Steve = human.get_activated_words()[1]
        word_dict = Steve.word_dict
        linear_Doug = STN.Dg(human.linear_corpus)
        if VERBOSE:
            Steve.plot_network()
        source = 'waiting'
        target = ['rabbit','deer','apple','water']
        testing = target
        testing.append(source)


        judge = check_word_in_list(testing, word_dict)

        recording_matrix = np.zeros((2*len(window_sizes) + 3, len(window_weights) * len(window_types)))
        subjective_matrix = np.zeros((2*len(window_sizes) + 3, len(window_weights) * len(window_types)))

        if judge:
            print('not met')
            break

        target.pop()

        p_nouns = target
        t_verbs = human.t_verb
        # print(p_nouns)
        # print(t_verbs)
        pairs = {}
        for noun in p_nouns:
            for verb in t_verbs:
                if (verb,noun) in human.t_p_pairs:
                    pairs[(verb,noun)] = human.t_p_pairs[(verb,noun)]
        ranking = np.zeros((len(p_nouns), len(t_verbs)))
        for phrase in pairs:
            id_argument = p_nouns.index(phrase[1])
            id_predicate = t_verbs.index(phrase[0])
            ranking[id_argument][id_predicate] = pairs[phrase]
        # print(ranking)
        standard_ranking = calculate_rank_matrix(ranking, 'standard')

        single_ranking = standard_ranking[t_verbs.index(source)]
        #print(single_ranking)
        num_correct = 0
        if single_ranking[0] > single_ranking[1] > single_ranking[2] > single_ranking[3]:
            num_correct = 1
        #else:
        #    print(ranking)


        sim_source = target[0]
        sim_target = target

        if hal:
            for i in range(len(window_sizes)):
                for j in range(len(window_weights)):
                    for k in range(len(window_types)):
                        encoding = {'window_size':window_sizes[i], 'window_weight':window_weights[j],
                                    'window_type':window_types[k]}

                        #sim_hal = HAL_analysis.get_cos_sim(linear_corpus,sim_source,sim_target,encoding, False)

                        sl_hal = HAL_analysis.get_cos_sim(linear_corpus,source,target,encoding, False)
                        if dict_to_rank(sl_hal) == str(single_ranking):
                            recording_matrix[2*i][j*len(window_types)+k] = 1
                        sim_hal = HAL_analysis.get_cos_sim(linear_corpus, sim_source, sim_target, encoding, False)
                        if dict_to_rank(sl_hal) == dict_to_rank(sim_hal):
                            subjective_matrix[2 * i][j * len(window_types) + k] = 1

                        sl_hal_svd = HAL_analysis.get_cos_sim(linear_corpus, source, target, encoding, True)
                        if dict_to_rank(sl_hal_svd) == str(single_ranking):
                            recording_matrix[2*i+1][j * len(window_types) + k] = 1
                        sim_hal_svd = HAL_analysis.get_cos_sim(linear_corpus, sim_source, sim_target, encoding, True)
                        if dict_to_rank(sl_hal_svd) == dict_to_rank(sim_hal_svd):
                            subjective_matrix[2 * i + 1][j * len(window_types) + k] = 1

                        if VERBOSE:
                            print(encoding)
                            print(sl_hal)
                            print(sl_hal_svd)
        if stv:
            sl_steve = STN_analysis.activation_spreading_analysis(Steve, source, target)
            if VERBOSE:
                print('semantic relatedness by STN:')
                print(sl_steve)
            if dict_to_rank(sl_steve) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][0] = 1

            sim_steve = STN_analysis.activation_spreading_analysis(Steve, sim_source, sim_target)
            if dict_to_rank(sl_steve) == dict_to_rank(sim_steve):
                subjective_matrix[2*len(window_sizes)][0] = 1


            reverse_target = [source,target[0]]
            reverse_relatedness = {}
            reverse_sim = {}
            for word in target:
                reverse_source = word
                relatedness = STN_analysis.activation_spreading_analysis(Steve, reverse_source, reverse_target)
                reverse_relatedness[word] = relatedness[reverse_target[0]]
                reverse_sim[word] = relatedness[reverse_target[1]]
            if dict_to_rank(reverse_relatedness) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][1] = 1
            if dict_to_rank(reverse_relatedness) == dict_to_rank(reverse_sim):
                subjective_matrix[2*len(window_sizes)][1] = 1


        if doug:
            sl_doug = STN_analysis.activation_spreading_analysis(linear_Doug, source, target)
            if VERBOSE:
                print('semantic relatedness by Distributional Graph')
                print(sl_doug)
            if dict_to_rank(sl_doug) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][2] = 1

            sim_doug = STN_analysis.activation_spreading_analysis(linear_Doug, sim_source, sim_target)
            if dict_to_rank(sl_doug) == dict_to_rank(sim_doug):
                subjective_matrix[2*len(window_sizes)][2] = 1

            reverse_target = [source, target[0]]
            reverse_relatedness = {}
            reverse_sim = {}
            for word in target:
                reverse_source = word
                relatedness = STN_analysis.activation_spreading_analysis(linear_Doug, reverse_source, reverse_target)
                reverse_relatedness[word]= relatedness[reverse_target[0]]
                reverse_sim[word] = relatedness[reverse_target[1]]
            if dict_to_rank(reverse_relatedness) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][3] = 1
            if dict_to_rank(reverse_relatedness) == dict_to_rank(reverse_sim):
                subjective_matrix[2*len(window_sizes)][3] = 1


        if synhal:
            window_weight = 'syntax'
            sl_synhal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
            if VERBOSE:
                print('semantic relatedness by Syntactic HAL')
                print(sl_synhal)
            if dict_to_rank(sl_synhal) == str(single_ranking):
                recording_matrix[2*len(window_sizes)+1][0] = 1

            sim_synhal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, sim_source, sim_target, window_weight, False)
            if dict_to_rank(sl_synhal) == dict_to_rank(sim_synhal):
                subjective_matrix[2*len(window_sizes)+1][0] = 1


            sl_synhal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
            if VERBOSE:
                print('semantic relatedness by Syntactic HAL after SVD')
                print(sl_synhal)
            if dict_to_rank(sl_synhal_svd) == str(single_ranking):
                recording_matrix[2*len(window_sizes) + 2][0] = 1

            sim_synhal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, sim_source, sim_target, window_weight,
                                                     True)
            if dict_to_rank(sl_synhal_svd) == dict_to_rank(sim_synhal_svd):
                subjective_matrix[2 * len(window_sizes) + 2][0] = 1

        if senthal:
            for window_weight in window_weights:
                sl_senthal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
                if VERBOSE:
                    print('semantic relatedness by {} Sentential HAL'.format(window_weight))
                    print(sl_senthal)
                if dict_to_rank(sl_senthal) == str(single_ranking):
                    recording_matrix[2*len(window_sizes)+1][window_weights.index(window_weight)+1] = 1

                sim_senthal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, sim_source, sim_target, window_weight,
                                                         False)
                if dict_to_rank(sl_senthal) == dict_to_rank(sim_senthal):
                    subjective_matrix[2 * len(window_sizes) + 1][window_weights.index(window_weight)+1] = 1


                sl_senthal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
                if VERBOSE:
                    print('semantic relatedness by {} Sentential HAL after SVD'.format(window_weight))
                    print(sl_senthal_svd)
                if dict_to_rank(sl_senthal_svd) == str(single_ranking):
                    recording_matrix[2*len(window_sizes) + 2][window_weights.index(window_weight) + 1] = 1

                sim_senthal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, sim_source, sim_target, window_weight,
                                                          True)
                if dict_to_rank(sl_senthal_svd) == dict_to_rank(sim_senthal_svd):
                    subjective_matrix[2 * len(window_sizes) + 2][window_weights.index(window_weight) + 1] = 1

        matrices.append(recording_matrix)
        matrices.append(subjective_matrix)

    return matrices, num_correct


def calculate_rank_matrix(matrix,version):
    transpose = matrix.transpose()
    (m,n)= matrix.shape
    rank_matrix = np.zeros((n,m))
    if version == 'standard':
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

    return rank_matrix


def ordering_task_analysis():
    the_world = running_world()
    matrices = []
    for human in the_world.human_list:
        corpus = human.corpus
        linear_corpus = human.linear_corpus
        Steve = human.get_activated_words()[1]
        linear_Doug = STN.Dg(human.linear_corpus)
        p_nouns = human.p_noun
        t_verbs = human.t_verb
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
        standard_ranking = calculate_rank_matrix(ranking,'standard')
        #print('standard')
        #print(standard_ranking)
        recording_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))

        if hal:
            for i in range(len(window_sizes)):
                for j in range(len(window_weights)):
                    for k in range(len(window_types)):
                        encoding = {'window_size':window_sizes[i], 'window_weight':window_weights[j],
                                    'window_type':window_types[k]}
                        hal_matrix = ranking
                        hal_svd_matrix = ranking
                        for source in t_verbs:
                            sl_hal = HAL_analysis.get_cos_sim(linear_corpus,source,target,encoding, False)
                            sl_hal_svd = HAL_analysis.get_cos_sim(linear_corpus, source, target, encoding, True)
                            for word in target:
                                id1 = p_nouns.index(word)
                                id2 = t_verbs.index(source)
                                hal_matrix[id1][id2] = sl_hal[word]
                                hal_svd_matrix[id1][id2] = sl_hal_svd[word]
                        hal_ranking = calculate_rank_matrix(hal_matrix,'non')
                        hal_svd_ranking = calculate_rank_matrix(hal_svd_matrix,'non')
                        corr_hal = np.corrcoef(hal_ranking.flatten(),standard_ranking.flatten())[0][1]
                        corr_hal_svd = np.corrcoef(hal_svd_ranking.flatten(),standard_ranking.flatten())[0][1]
                        recording_matrix[2 * i][j * len(window_types) + k] = corr_hal
                        recording_matrix[2 * i + 1][j * len(window_types) + k] = corr_hal_svd


        if stv:
            stv_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            stv_re_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            for source in t_verbs:
                sl_steve = STN_analysis.activation_spreading_analysis(Steve, source, target)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    stv_matrix[id1][id2] = sl_steve[word]
            stv_ranking = calculate_rank_matrix(stv_matrix, 'non')
            #print('STN')
            #print(stv_ranking)

            re_target = t_verbs
            for re_source in p_nouns:
                sl_re_steve = STN_analysis.activation_spreading_analysis(Steve, re_source, re_target)
                for word in re_target:
                    id2 = t_verbs.index(word)
                    id1 = p_nouns.index(re_source)
                    stv_re_matrix[id1][id2] = sl_re_steve[word]

            stv_re_ranking = calculate_rank_matrix(stv_re_matrix,'non')
            #print('STN reversed')
            #print(stv_re_ranking)
            corr_stv = np.corrcoef(stv_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_stv_re = np.corrcoef(stv_re_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes)][0] = corr_stv
            recording_matrix[2*len(window_sizes)][1] = corr_stv_re


        if doug:
            doug_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            doug_re_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            for source in t_verbs:
                sl_doug = STN_analysis.activation_spreading_analysis(linear_Doug, source, target)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    doug_matrix[id1][id2] = sl_doug[word]
            doug_ranking = calculate_rank_matrix(doug_matrix, 'non')
            #print('distributional graph')
            #print(doug_ranking)
            re_target = t_verbs
            for re_source in p_nouns:
                sl_re_doug = STN_analysis.activation_spreading_analysis(linear_Doug, re_source, re_target)
                for word in re_target:
                    id2 = t_verbs.index(word)
                    id1 = p_nouns.index(re_source)
                    doug_re_matrix[id1][id2] = sl_re_doug[word]


            doug_re_ranking = calculate_rank_matrix(doug_re_matrix, 'non')
            #print('distributional graph reversed')
            #print(doug_re_ranking)
            corr_doug = np.corrcoef(doug_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_doug_re = np.corrcoef(doug_re_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes)][2] = corr_doug
            recording_matrix[2*len(window_sizes)][3] = corr_doug_re

        if synhal:
            synhal_matrix = ranking
            synhal_svd_matrix = ranking
            window_weight = 'syntax'
            for source in t_verbs:
                sl_synhal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
                sl_synhal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    synhal_matrix[id1][id2] = sl_synhal[word]
                    synhal_svd_matrix[id1][id2] = sl_synhal_svd[word]
            synhal_ranking = calculate_rank_matrix(synhal_matrix, 'non')
            synhal_svd_ranking = calculate_rank_matrix(synhal_svd_matrix, 'non')
            corr_synhal = np.corrcoef(synhal_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_synhal_svd = np.corrcoef(synhal_svd_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes) + 1][0] = corr_synhal
            recording_matrix[2*len(window_sizes) + 2][0] = corr_synhal_svd


        if senthal:
            for window_weight in window_weights:
                senthal_matrix = ranking
                senthal_svd_matrix = ranking
                for source in t_verbs:
                    sl_senthal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
                    sl_senthal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
                    for word in target:
                        id1 = p_nouns.index(word)
                        id2 = t_verbs.index(source)
                        senthal_matrix[id1][id2] = sl_senthal[word]
                        senthal_svd_matrix[id1][id2] = sl_senthal_svd[word]
                senthal_ranking = calculate_rank_matrix(senthal_matrix, 'non')
                senthal_svd_ranking = calculate_rank_matrix(senthal_svd_matrix, 'non')
                corr_senthal = np.corrcoef(senthal_ranking.flatten(), standard_ranking.flatten())[0][1]
                corr_senthal_svd = np.corrcoef(senthal_svd_ranking.flatten(), standard_ranking.flatten())[0][1]
                recording_matrix[2*len(window_sizes) + 1][window_weights.index(window_weight) + 1] = corr_senthal
                recording_matrix[2*len(window_sizes) + 2][window_weights.index(window_weight) + 1] = corr_senthal_svd

        matrices.append(recording_matrix)
    return matrices


def run_experiments(run_times,experiment):
    objective_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    subjective_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    objective_count = 0
    for i in range(run_times):
        if experiment == 'one_task':
            a,b = one_ordering_task()
            objective_matrix += a[0]
            subjective_matrix += a[1]
            objective_count += b
        else:
            objective_matrix += ordering_task_analysis()[0]
        if i % 5 == 0:
            print('{} turns run'.format(i))
    objective_matrix = objective_matrix/run_times
    subjective_matrix = subjective_matrix/run_times
    objective_rate = objective_count/run_times
    print(objective_matrix)
    print(subjective_matrix)
    print(objective_rate)


run_experiments(10,'one_task')