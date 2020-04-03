from Programs.World import world, config
from Programs.Spatial_Models import spacial_analysis
from Programs.Graphical_Models import STN, synHAL_analysis, graphical_analysis
from Programs.Experiment import paradigmatic_task as p_task
from Programs.Experiment import syntagmatic_task as s_task
from pathlib import Path
import numpy as np
import scipy.stats as ss
import csv
import matplotlib.pyplot as plt
import math

VERBOSE = False
window_types = ['forward','summed']
window_sizes = [9]
window_weights = ['linear','flat']
exp_length = 1

stv = False
doug = True
hal = True
synhal = False
senthal = False

def get_num_model():
    num_model = 0
    if stv:
        num_model = num_model + 2
    if doug:
        num_model = num_model + 2
    if synhal:
        num_model = num_model + 2
    if senthal:
        num_model = num_model + 2 * len(window_weights)
    if hal:
        num_model = num_model + 2 * len(window_weights) * len(window_sizes) * len(window_types)
    return num_model



def check_word_in_list(the_list, the_dict):
    judge = 0
    for word in the_list:
        if word not in the_dict:
            judge = judge + 1
            break
    return judge


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


def dict_to_rank(dict):
    list = []
    for word in dict:
        list.append(dict[word])
    rank = str(ss.rankdata(list))
    return rank

########################################################################################################################
# objective and subjective one-ordering task
# the semantic models have to rank the syntagmatic relatedness between one verb(trap) vs several nouns
# (rabbit,deer,apple,water), and the ranking is evaluated by comparing to an objective key(objective task)or a
# subjective key(subjective task)

# the key of objective task is obtained by manually binding the paradigmatic relation and the syntagmatic relation
# basing on the corpus

# the key of subjective task is the models' ranking of the paradigmatic relations

# all spatial models are tested with, or without svd transformation
# all graphical models are tested with two direction semantic relatedness, notice that both Re(trap, rabbit), and
# Re(rabbit, trap) tell something about the relatedness between trap and rabbit, currently, there are empirical studies
# showing that the relations is asymmetric, and which is also true in the measure implementation, meanwhile, there is no
# theoretical account for one direction precedes over the other, thus we use both directions in the relatedness measures
# that is, when comparing the relatedness between trap with rabbit and apple, we compare from both directions:
# Re(trap, rabbit) vs Re(trap, deer), adn also Re(rabbit, trap) vs Re(deer, trap).

# graphical models use activation-spreading to measure the semantic relatedness, while spatial models compute the
# cosine similarity from the ppmi matrix.

########################################################################################################################


def one_ordering_task():
    the_world = running_world()
    matrices = []
    for human in the_world.human_list:
        # print('eat fruit {}'.format(human.eat_count_fruit))
        # print('eat meal {}'.format(human.eat_count_meal))
        # print('drink {}'.format(human.drink_count))
        # print('sleep {}'.format(human.sleep_count))
        # print('idle {}'.format(human.idle_count))
        corpus = human.corpus
        linear_corpus = human.linear_corpus
        steve = human.get_activated_words()[1]
        word_dict = steve.word_dict
        linear_Doug = STN.Dg(human.linear_corpus)
        if VERBOSE:
            steve.plot_network()
        source = 'trapping'
        target = ['rabbit','deer','apple','water']  # the nouns, where rabbit co-occur with the source verb, others
        # don't, but has closer or farther paradigmatic relatedness rabbit
        testing = target
        testing.append(source)
        judge = check_word_in_list(testing, word_dict) # to see if the targets and the source are in the corpus

        # generating the matrix to record evaluation scores
        recording_matrix = np.zeros((2*len(window_sizes) + 3, len(window_weights) * len(window_types)))
        subjective_matrix = np.zeros((2*len(window_sizes) + 3, len(window_weights) * len(window_types)))

        if judge:
            print('not met')
            matrices.append(recording_matrix)
            matrices.append(subjective_matrix)
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
        standard_ranking = s_task.calculate_rank_matrix(ranking, 'standard') # get the objective key

        single_ranking = standard_ranking[t_verbs.index(source)]
        # print(single_ranking)

        # else:
        #    print(ranking)

        sim_source = target[0]
        sim_target = target

        if hal:
            for i in range(len(window_sizes)):
                for j in range(len(window_weights)):
                    for k in range(len(window_types)):
                        encoding = {'window_size':window_sizes[i], 'window_weight':window_weights[j],
                                    'window_type':window_types[k]}

                        sl_hal = sim_space_analysis.get_cos_sim(linear_corpus, source, target, encoding, False)
                        if dict_to_rank(sl_hal) == str(single_ranking):
                            recording_matrix[2*i][j*len(window_types)+k] = 1
                        sim_hal = sim_space_analysis.get_cos_sim(linear_corpus, sim_source, sim_target, encoding, False)
                        if dict_to_rank(sl_hal) == dict_to_rank(sim_hal):
                            subjective_matrix[2 * i][j * len(window_types) + k] = 1

                        sl_hal_svd = sim_space_analysis.get_cos_sim(linear_corpus, source, target, encoding, True)
                        if dict_to_rank(sl_hal_svd) == str(single_ranking):
                            recording_matrix[2*i+1][j * len(window_types) + k] = 1
                        sim_hal_svd = sim_space_analysis.get_cos_sim(linear_corpus, sim_source, sim_target, encoding, True)
                        if dict_to_rank(sl_hal_svd) == dict_to_rank(sim_hal_svd):
                            subjective_matrix[2 * i + 1][j * len(window_types) + k] = 1

                        if VERBOSE:
                            print(encoding)
                            print(sl_hal)
                            print(sl_hal_svd)
        if stv:
            sl_steve = graphical_analysis.activation_spreading_analysis(steve, source, target)
            if VERBOSE:
                print('semantic relatedness by STN:')
                print(sl_steve)
            if dict_to_rank(sl_steve) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][0] = 1

            sim_steve = graphical_analysis.activation_spreading_analysis(steve, sim_source, sim_target)
            if dict_to_rank(sl_steve) == dict_to_rank(sim_steve):
                subjective_matrix[2*len(window_sizes)][0] = 1
            reverse_target = [source,target[0]]
            reverse_relatedness = {}
            reverse_sim = {}
            for word in target:
                reverse_source = word
                relatedness = graphical_analysis.activation_spreading_analysis(steve, reverse_source, reverse_target)
                reverse_relatedness[word] = relatedness[reverse_target[0]]
                reverse_sim[word] = relatedness[reverse_target[1]]
            if dict_to_rank(reverse_relatedness) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][1] = 1
            if dict_to_rank(reverse_relatedness) == dict_to_rank(reverse_sim):
                subjective_matrix[2*len(window_sizes)][1] = 1

        if doug:
            sl_doug = graphical_analysis.activation_spreading_analysis(linear_Doug, source, target)
            if VERBOSE:
                print('semantic relatedness by Distributional Graph')
                print(sl_doug)
            if dict_to_rank(sl_doug) == str(single_ranking):
                recording_matrix[2*len(window_sizes)][2] = 1

            sim_doug = graphical_analysis.activation_spreading_analysis(linear_Doug, sim_source, sim_target)
            if dict_to_rank(sl_doug) == dict_to_rank(sim_doug):
                subjective_matrix[2*len(window_sizes)][2] = 1

            reverse_target = [source, target[0]]
            reverse_relatedness = {}
            reverse_sim = {}
            for word in target:
                reverse_source = word
                relatedness = graphical_analysis.activation_spreading_analysis(linear_Doug, reverse_source, reverse_target)
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

    return matrices







########################################################################################################################
# generalized objective (syntagmatic) ranking task & paradigmatic categorization task


# generalized objective (syntagmatic) ranking task:

# similar to the one-ordering task, the generalized task generalize the ranking task to all verbs in the corpus, and for
# each verb, it is generalized to the ranking over all nouns.
# for each verb, all nouns are ranked with respective to their syntagmatic relatedness to that verb
# the objective ranking is formed by first ranking the noun(s) having syntagmatic relations to the verb by co-occur
# frequency and then rank the nouns without syntagmatic relations after the co-occurred nouns, by their mean similarity
# (paradigmatic relatedness) to the co-occurred nouns
# each model carry out the semantic relatedness task and get the rankings for all verbs, and then correlated to the
# objective ranking

# paradigmatic categorization task:
# each model measures the semantic relatedness between all noun pairs, and collapsing the similarities into
# 'within_category' and 'between_category' groups, where the categories are ad hoc noun categories like 'animal'
# 'fruit', and 'drink'
########################################################################################################################

def ordering_task_analysis():
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
        recording_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
        data_matrix = flat_standard
        category_sim = []
        within_between = []

        if hal:
            for i in range(len(window_sizes)):
                for j in range(len(window_weights)):
                    for k in range(len(window_types)):
                        encoding = {'window_size':window_sizes[i], 'window_weight':window_weights[j],
                                    'window_type':window_types[k]}
                        hal_matrix = np.zeros((len(p_nouns),len(t_verbs)))
                        hal_svd_matrix = np.zeros((len(p_nouns),len(t_verbs)))

                        for source in t_verbs:
                            sl_hal = sim_space_analysis.get_cos_sim(linear_corpus, source, target, encoding, False)
                            sl_hal_svd = sim_space_analysis.get_cos_sim(linear_corpus, source, target, encoding, True)
                            for word in target:
                                id1 = p_nouns.index(word)
                                id2 = t_verbs.index(source)
                                hal_matrix[id1][id2] = sl_hal[word]
                                hal_svd_matrix[id1][id2] = sl_hal_svd[word]
                        hal_ranking = s_task.calculate_rank_matrix(hal_matrix,'non')
                        hal_svd_ranking = s_task.calculate_rank_matrix(hal_svd_matrix,'non')
                        flat_hal = hal_ranking.flatten().reshape((rank_size,1))
                        flat_hal_svd = hal_svd_ranking.flatten().reshape((rank_size,1))
                        data_matrix = np.concatenate((data_matrix,flat_hal),1)
                        data_matrix = np.concatenate((data_matrix,flat_hal_svd),1)
                        corr_hal = np.corrcoef(hal_ranking.flatten(),standard_ranking.flatten())[0][1]
                        corr_hal_svd = np.corrcoef(hal_svd_ranking.flatten(),standard_ranking.flatten())[0][1]
                        recording_matrix[2 * i][j * len(window_types) + k] = corr_hal
                        recording_matrix[2 * i + 1][j * len(window_types) + k] = corr_hal_svd
                        category_sim_matrix, within_between_dict = p_task.get_category_sim(p_nouns, the_world, None,
                                                                                    linear_corpus, encoding,
                                                                                    'hal', False)
                        category_sim.append(category_sim_matrix)
                        within_between.append(within_between_dict)


        if stv:
            stv_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            stv_re_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            for source in t_verbs:
                sl_steve = graphical_analysis.activation_spreading_analysis(Steve, source, target)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    stv_matrix[id1][id2] = sl_steve[word]
            stv_ranking = s_task.calculate_rank_matrix(stv_matrix, 'non')
            #print('STN')
            #print(stv_ranking)

            re_target = t_verbs
            for re_source in p_nouns:
                sl_re_steve = graphical_analysis.activation_spreading_analysis(Steve, re_source, re_target)
                for word in re_target:
                    id2 = t_verbs.index(word)
                    id1 = p_nouns.index(re_source)
                    stv_re_matrix[id1][id2] = sl_re_steve[word]

            stv_re_ranking = s_task.calculate_rank_matrix(stv_re_matrix,'non')
            flat_stv = stv_ranking.flatten().reshape((rank_size, 1))
            flat_stv_re = stv_re_ranking.flatten().reshape((rank_size, 1))
            data_matrix = np.concatenate((data_matrix, flat_stv), 1)
            data_matrix = np.concatenate((data_matrix, flat_stv_re), 1)
            #print('STN reversed')
            #print(stv_re_ranking)
            corr_stv = np.corrcoef(stv_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_stv_re = np.corrcoef(stv_re_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes)][0] = corr_stv
            recording_matrix[2*len(window_sizes)][1] = corr_stv_re
            category_sim_matrix, within_between_dict = p_task.get_category_sim(p_nouns, the_world, Steve, None, None,
                                                   'doug', None)
            category_sim.append(category_sim_matrix)
            within_between.append(within_between_dict)


        if doug:
            doug_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            doug_re_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            for source in t_verbs:
                sl_doug = graphical_analysis.activation_spreading_analysis(linear_Doug, source, target)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    doug_matrix[id1][id2] = sl_doug[word]
            doug_ranking = s_task.calculate_rank_matrix(doug_matrix, 'non')
            #print('distributional graph')
            #print(doug_ranking)
            re_target = t_verbs
            for re_source in p_nouns:
                sl_re_doug = graphical_analysis.activation_spreading_analysis(linear_Doug, re_source, re_target)
                for word in re_target:
                    id2 = t_verbs.index(word)
                    id1 = p_nouns.index(re_source)
                    doug_re_matrix[id1][id2] = sl_re_doug[word]


            doug_re_ranking = s_task.calculate_rank_matrix(doug_re_matrix, 'non')
            flat_doug = doug_ranking.flatten().reshape((rank_size, 1))
            flat_doug_re = doug_re_ranking.flatten().reshape((rank_size, 1))
            data_matrix = np.concatenate((data_matrix, flat_doug), 1)
            data_matrix = np.concatenate((data_matrix, flat_doug_re), 1)
            #print('distributional graph reversed')
            #print(doug_re_ranking)
            corr_doug = np.corrcoef(doug_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_doug_re = np.corrcoef(doug_re_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes)][2] = corr_doug
            recording_matrix[2*len(window_sizes)][3] = corr_doug_re
            category_sim_matrix, within_between_dict = p_task.get_category_sim(p_nouns, the_world, linear_Doug, None, None,
                                                        'doug', None)
            category_sim.append(category_sim_matrix)
            within_between.append(within_between_dict)

        if synhal:
            synhal_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            synhal_svd_matrix = np.zeros((len(p_nouns),len(t_verbs)))
            window_weight = 'syntax'
            for source in t_verbs:
                sl_synhal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
                sl_synhal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
                for word in target:
                    id1 = p_nouns.index(word)
                    id2 = t_verbs.index(source)
                    synhal_matrix[id1][id2] = sl_synhal[word]
                    synhal_svd_matrix[id1][id2] = sl_synhal_svd[word]
            synhal_ranking = s_task.calculate_rank_matrix(synhal_matrix, 'non')
            synhal_svd_ranking = s_task.calculate_rank_matrix(synhal_svd_matrix, 'non')
            flat_synhal = synhal_ranking.flatten().reshape((rank_size, 1))
            flat_synhal_svd = synhal_svd_ranking.flatten().reshape((rank_size, 1))
            data_matrix = np.concatenate((data_matrix, flat_synhal), 1)
            data_matrix = np.concatenate((data_matrix, flat_synhal_svd), 1)
            corr_synhal = np.corrcoef(synhal_ranking.flatten(), standard_ranking.flatten())[0][1]
            corr_synhal_svd = np.corrcoef(synhal_svd_ranking.flatten(), standard_ranking.flatten())[0][1]
            recording_matrix[2*len(window_sizes) + 1][0] = corr_synhal
            recording_matrix[2*len(window_sizes) + 2][0] = corr_synhal_svd
            category_sim_matrix, within_between_dict = p_task.get_category_sim(p_nouns, the_world, corpus, linear_corpus,
                                                                        window_weight,'spatial', False)
            category_sim.append(category_sim_matrix)
            within_between.append(within_between_dict)

        if senthal:
            for window_weight in window_weights:
                senthal_matrix = np.zeros((len(p_nouns),len(t_verbs)))
                senthal_svd_matrix = np.zeros((len(p_nouns),len(t_verbs)))
                for source in t_verbs:
                    sl_senthal = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, False)
                    sl_senthal_svd = synHAL_analysis.get_cos_sim(corpus, linear_corpus, source, target, window_weight, True)
                    id2 = t_verbs.index(source)
                    for word in target:
                        id1 = p_nouns.index(word)
                        senthal_matrix[id1][id2] = sl_senthal[word]
                        senthal_svd_matrix[id1][id2] = sl_senthal_svd[word]

                senthal_ranking = s_task.calculate_rank_matrix(senthal_matrix, 'non')
                senthal_svd_ranking = s_task.calculate_rank_matrix(senthal_svd_matrix, 'non')
                flat_senthal = senthal_ranking.flatten().reshape((rank_size, 1))
                flat_senthal_svd = senthal_svd_ranking.flatten().reshape((rank_size, 1))
                data_matrix = np.concatenate((data_matrix, flat_senthal), 1)
                data_matrix = np.concatenate((data_matrix, flat_senthal_svd), 1)
                corr_senthal = np.corrcoef(senthal_ranking.flatten(), standard_ranking.flatten())[0][1]
                corr_senthal_svd = np.corrcoef(senthal_svd_ranking.flatten(), standard_ranking.flatten())[0][1]
                recording_matrix[2*len(window_sizes) + 1][window_weights.index(window_weight) + 1] = corr_senthal
                recording_matrix[2*len(window_sizes) + 2][window_weights.index(window_weight) + 1] = corr_senthal_svd
                category_sim_matrix, within_between_dict = p_task.get_category_sim(p_nouns, the_world, corpus, linear_corpus,
                                                                            window_weight,'spatial', False)
                category_sim.append(category_sim_matrix)
                within_between.append(within_between_dict)

        matrices.append((recording_matrix,data_matrix,flat_item,category_sim, within_between, num_sentence))

    return matrices


def bar_graph(group1,group2,bar_width,x_ticks,y_label):
    bars1 = group1[0]
    yer1 = group1[1]
    bars2 = group2[0]
    yer2 = group2[1]
    r1 = np.arange(len(bars1))
    r2 = [x + bar_width for x in r1]
    plt.bar(r1, bars1, width=bar_width, color='blue', edgecolor='black', yerr=yer1, capsize=7, label=group1[2])
    plt.bar(r2, bars2, width=bar_width, color='cyan', edgecolor='black', yerr=yer2, capsize=7, label=group2[2])
    plt.xticks([r + bar_width/2 for r in range(len(bars1))], x_ticks)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

########################################################################################################################
# running objective (and subjective) ranking task

########################################################################################################################


def run_experiments(length,experiment):
    objective_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    subjective_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    objective_count = 0
    for i in range(length):
        if experiment == 'one_task':
            a = one_ordering_task()
            objective_matrix += a[0]
            subjective_matrix += a[1]
        else:
            objective_matrix += ordering_task_analysis()[0][0]
        if i % 5 == 0:
            print('{} turns run'.format(i))
    objective_matrix = objective_matrix/length
    subjective_matrix = subjective_matrix/length
    objective_rate = objective_count/length
    print(objective_matrix)
    # print(subjective_matrix)
    # print(objective_rate)

########################################################################################################################
# running generalized objective ranking task (multiple runs)
# for each model tested in the task, average correlation score (and SE) are caculated, bargraph plotted.

########################################################################################################################


def run_experiments_order(length):
    num = get_num_model()
    objective_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    category_sim = []
    within_between = []
    corr_spatial = []
    corr_graphical = []
    num_sentence = []
    for i in range(num):
        category_sim.append( np.zeros((3,3)))
        within_between.append([])
    path = Path().cwd().parent / 'Data' / 'ranking.csv'
    with path.open('w') as csvfile: # save the standard ranking and model rankings in a csv file.
        fieldnames = ['Subject','Item','Key']
        for i in range(num):
            model = 'M'+str(i)
            fieldnames.append(model)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(length):
            if i % 5 == 0:
                print('{} turns run'.format(i))
            corr_matrix, next_matrix, item, sim, w_b, num_sent= ordering_task_analysis()[0]
            corr_spatial.append(corr_matrix[2*len(window_sizes)-2][len(window_types)-1])
            corr_graphical.append(corr_matrix[2*len(window_sizes)][2])
            objective_matrix += corr_matrix
            num_sentence.append(num_sent)
            if i == 0:
                category_sim = sim
                within_between = w_b
            num_row = next_matrix.shape[0]
            for j in range(num_row):
                row = {}
                for k in range(num+1):
                    row[fieldnames[k+2]] = next_matrix[j][k]
                row['Subject'] = i + 1
                row['Item'] = item[j]
                writer.writerow(row)
        objective_matrix = objective_matrix/length
        num_sentence = (np.array(num_sentence).mean(),np.array(num_sentence).std())
        print(num_sentence)
        print(objective_matrix)
        print('category_sim')
        print()
        for matrix in category_sim:
            print(matrix)
            print()
        within = [[within_between[1][0][0],within_between[-1][0][0]],[within_between[1][0][1],
                                                                         within_between[-1][0][1]],'within category']

        between = [[within_between[1][1][0],within_between[-1][1][0]],[within_between[1][1][1],
                                                                         within_between[-1][1][1]],'between category']
        bar_graph(within,between,0.3,['spatial','graphical'],'relatedness')
        for w_b in within_between:
            print(w_b)
            print()
        corr_g = np.array(corr_graphical)
        corr_s = np.array(corr_spatial)
        spatial = [[corr_s.mean()],[corr_s.std()/math.sqrt(length)],'spatial']
        graphical = [[corr_g.mean()],[corr_g.std()/math.sqrt(length)],'graphical']
        bar_graph(spatial,graphical,0.05,[' '],'correlation')
        print(spatial)
        print(graphical)

    # np.savetxt("performance.csv", objective_matrix, delimiter=",")


run_experiments_order(exp_length)
