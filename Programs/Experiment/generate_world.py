from Programs.World import world, config
import csv
import numpy as np

VERBOSE = True


def running_world():  # running the world and get the corpus
    the_world = world.World()
    the_world.create_fruit()
    the_world.create_nut()
    the_world.create_plant()
    the_world.create_herbivores()
    the_world.create_humans()
    the_world.create_carnivores()

    for i in range(config.World.num_turn):
        the_world.next_turn()
    for agent in the_world.agent_list:
        the_world.eat_count_nut += agent.eat_count_nut
        the_world.eat_count_fruit += agent.eat_count_fruit
        the_world.eat_count_meal += agent.eat_count_meal
        the_world.eat_count_plant += agent.eat_count_plant
        the_world.sleep_count += agent.sleep_count
        the_world.idle_count += agent.idle_count
        the_world.drink_count += agent.drink_count
    # the_display = display.Display(the_world)
    # the_display.root.mainloop()
    num_consumed_herbivore = config.World.num_herbivores * len(the_world.herbivore_category) - len(the_world.herbivore_list)
    sorted_t_p_pairs = sorted(the_world.t_p_pairs.items(),key=lambda x:x[1], reverse=True)
    sorted_v_a_pairs = sorted(the_world.v_a_pairs.items(), key=lambda x: x[1], reverse=True)
    for pair in the_world.v_a_pairs:
        the_world.pairs[pair] = the_world.v_a_pairs[pair]
    for pair in the_world.t_p_pairs:
        if pair in the_world.pairs:
            the_world.pairs[pair] += the_world.t_p_pairs[pair]
        else:
            the_world.pairs[pair] = the_world.t_p_pairs[pair]
    length = len(the_world.corpus)
    l_verb = len(the_world.verb)
    l_t_verb = len(the_world.t_verb)
    l_agent = len(the_world.agent)
    l_p_noun = len(the_world.p_noun)
    if VERBOSE:
        print('{} sentences'.format(length))
        print('verbs: {}'.format(the_world.verb))
        print('transitive verbs: {}'.format(the_world.t_verb))
        print('agents: {}'.format(the_world.agent))
        print('patients: {}'.format(the_world.p_noun))
        print('noun stems: {}'.format(the_world.noun_stems))
        print('noun dict: {}'.format(the_world.noun_dict))
        print('{} animals consumed.'.format(num_consumed_herbivore))
        print('{} epochs passed'.format(the_world.epoch))
        print('{} eating meals'.format(the_world.eat_count_meal))
        print('{} eating nuts'.format(the_world.eat_count_nut))
        print('{} eating fruit'.format(the_world.eat_count_fruit))
        print('{} drinking'.format(the_world.drink_count))
        print('{} sleeping'.format(the_world.sleep_count))
        print('{} human eat'.format(the_world.human_eat))
        print('{} carnivore eat'.format(the_world.carnivore_eat))
        print('{} herbivore eat'.format(the_world.herbivore_eat))
        for pairs in sorted_t_p_pairs:
            print(pairs)
        for pairs in sorted_v_a_pairs:
            print(pairs)
    return the_world

def get_syntagmatic_rule():
    syntagmatic_rule = {}
    verb_indices = {}
    with open('Experiment/Syntagmatic_rule.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                column = 0
                for item in row[1:]:
                    if item not in verb_indices:
                        verb_indices[item] = column
                    column += 1
            else :
                basic_category = row[0]
                copy = []
                for item in row[1:]:
                    copy.append(eval(item))
                syntagmatic_rule[basic_category] = np.asarray(copy)
            line = line + 1
    return syntagmatic_rule, verb_indices



def get_world_info(the_world):
    # there could be multiple humans in the world, in the current model, there is only one
    profiles = []  # a record of information with respect to individuals: individuals are keys, and values are the
    # experiment-relevant information regarding the individuals

    profile = {}  # currently a paradigmatic and a syntagmatic task are carried out, need to pass the input to the
    # task.
    output_info = {}
    kit = {} # a kit is the package of data in order to carrying out the tasks: including the
    # corpus, the verbs and nouns, and the verb-noun pairs and so on

    ###############################################################################################################
    # followings are preparation for the s_task and p_task in linear models, get the s_kit, p_kit for the tasks.
    ###############################################################################################################

    linear_corpus = the_world.linear_corpus
    p_nouns = the_world.p_noun
    agent = the_world.agent
    t_verbs = the_world.t_verb
    verbs = the_world.verb
    v_a_pairs = the_world.v_a_pairs
    t_p_pairs = the_world.t_p_pairs
    nouns = list(set(agent).union(set(p_nouns)))
    pairs = the_world.pairs
    noun_dict = the_world.noun_dict
    noun_stems = the_world.noun_stems
    noun_tax = the_world.noun_tax
    rules, verb_indices = get_syntagmatic_rule()

    #word_bag, vocab_list, vocab_index_dict = build_models.corpus_transformation(linear_corpus)
    #cooc_matrix, sim_matrix = build_models.build_model(word_bag, vocab_list, vocab_index_dict, model_parameters)
    flat_item = []
    for verb in t_verbs:
        for noun in p_nouns:
            phrase = verb + '_' + noun
            flat_item.append(phrase)


    #print(p_nouns)
    #print(t_verbs)
    kit['p_nouns'] = p_nouns
    kit['t_verbs'] = t_verbs
    kit['nouns'] = nouns
    kit['agent'] = agent
    kit['verbs'] = verbs
    kit['v_a_pairs'] = v_a_pairs
    kit['t_p_pairs'] = t_p_pairs
    kit['pairs'] = pairs
    kit['flat_item'] = flat_item
    kit['the_world'] = the_world
    kit['noun_dict'] = noun_dict
    kit['noun_stems'] = noun_stems
    kit['rules'] = rules
    kit['noun_tax'] = noun_tax
    kit['verb_indices'] = verb_indices

    ###############################################################################################################
    # followings are preparation for the s_task and p_task in syntactic models, get the s_kit, p_kit for the tasks.
    ###############################################################################################################
    # Steve = agent.get_activated_words()[1]
    corpus = the_world.corpus
    num_sentence = len(corpus)
    #flat_standard = standard_ranking.flatten().reshape(rank_size,1)

    #print('standard')
    #print(standard_ranking)
    #recording_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
    #data_matrix = flat_standard
    profile['kit'] = kit
    profile['output_info'] = output_info
    profile['num_sentence'] = num_sentence

    profiles.append(profile)
    return profiles, linear_corpus