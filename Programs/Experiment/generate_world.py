from Programs.World import world, config
import csv
import numpy as np

VERBOSE = True


def running_world():  # running the world and get the corpus
    the_world = world.World()
    the_world.create_fruit()
    the_world.create_nut()
    the_world.create_plant()
    for i in range(config.World.num_food_herbivores):
        the_world.create_food_herbivores()
    the_world.create_acting_herbivores()
    the_world.create_humans()
    the_world.create_carnivores()

    for i in range(config.World.num_turn):
        the_world.next_turn()

    # get evaluation similarity: whether carnivores are more similar to humans (on agent part)

    animal_list = []
    evaluate_dict = {}
    for c in the_world.carnivore_list:
        carnivore = c.category + '-a'
        animal_list.append(carnivore)
        evaluate_dict[carnivore] = []

    for i in range(6): # get one of each herbivore types
        herbivore = the_world.acting_herbivore_list[i].category + '-a'
        animal_list.append(herbivore)
        evaluate_dict[herbivore]=[]
    human_list = []

    for h in the_world.human_list:
        human = h.name + '-a'
        human_list.append(human)
        evaluate_dict[human] = []

    for agent in evaluate_dict:
        for v in the_world.verb:
            if (v, agent) in the_world.v_a_pairs:
                evaluate_dict[agent].append(the_world.v_a_pairs[(v,agent)])
            else:
                evaluate_dict[agent].append(0)
        evaluate_dict[agent] = np.asarray(evaluate_dict[agent])


    for animal in animal_list: # animals are ordered by their types, first carnivores, and then different types of herbivores
        v1 = evaluate_dict[animal]
        sim = 0
        for human in human_list:
            v2 = evaluate_dict[human]
            sim = sim + np.inner(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
        sim = sim/len(human_list)
        print(animal,sim)



    # count how many times agent eat certain food
    for agent in the_world.agent_list:
        the_world.eat_count_nut += agent.eat_count_nut
        the_world.eat_count_fruit += agent.eat_count_fruit
        the_world.eat_count_meal += agent.eat_count_meal
        the_world.eat_count_plant += agent.eat_count_plant
        the_world.sleep_count += agent.sleep_count
        the_world.idle_count += agent.idle_count
        the_world.drink_count += agent.drink_count

    for carnivore in the_world.carnivore_list:
        the_world.hunt_success += carnivore.hunt_success

    # the_display = display.Display(the_world)
    # the_display.root.mainloop()

    # count how many herbivores have been consumed
    num_consumed_herbivore = config.World.num_food_herbivores * len(the_world.herbivore_category) - len(the_world.herbivore_list)

    sorted_t_p_pairs = sorted(the_world.t_p_pairs.items(),key=lambda x:x[1], reverse=True)
    sorted_v_a_pairs = sorted(the_world.v_a_pairs.items(), key=lambda x: x[1], reverse=True)
    for pair in the_world.v_a_pairs:
        the_world.pairs[pair] = the_world.v_a_pairs[pair]

    for pair in the_world.t_p_pairs:
        if pair in the_world.pairs:
            the_world.pairs[pair] += the_world.t_p_pairs[pair]
        else:
            the_world.pairs[pair] = the_world.t_p_pairs[pair]

    for pair in the_world.pairs:
        collapsed_pair = (pair[0], pair[1][:-2])
        if collapsed_pair in the_world.collapsed_pairs:
            the_world.collapsed_pairs[collapsed_pair] += the_world.pairs[pair]
        else:
            the_world.collapsed_pairs[collapsed_pair] = the_world.pairs[pair]

    sorted_collapsed_pairs = sorted(the_world.collapsed_pairs.items(), key=lambda x: x[1], reverse=True)

    length = len(the_world.corpus)
    num_token = 0
    for sent in the_world.linear_corpus:
        num = len(sent)
        num_token = num_token + num

    l_verb = len(the_world.verb)
    l_t_verb = len(the_world.t_verb)
    l_agent = len(the_world.agent)
    l_p_noun = len(the_world.p_noun)

    num_pairs = len(the_world.v_a_pairs) + len(the_world.t_p_pairs)
    num_collapsed_pairs = len(the_world.collapsed_pairs)
    num_possible_pairs = l_agent * l_verb + l_t_verb * l_p_noun
    num_possible_collapsed = len(the_world.noun_stems) * l_verb

    rate_pairs = round(100 * num_pairs/num_possible_pairs,1)
    rate_collapsed_pairs = round(100 * num_collapsed_pairs/num_possible_pairs,1)

    print('{} sentences'.format(length))
    print('{} word tokens'.format(num_token))
    print('{}, or {}% of possible pairs has actually occurred'.format(num_pairs, rate_pairs))
    print('{}, or {}% of possible collapsed pairs has actually occurred'.format(num_collapsed_pairs,
                                                                                rate_collapsed_pairs))

    if VERBOSE:
        #print('First 100 sentences')
        #print()
        #for sentence in the_world.linear_corpus[:100]:
            #print(sentence)
        print()
        print('verbs: {}'.format(the_world.verb))
        print('transitive verbs: {}'.format(the_world.t_verb))
        print('agents: {}'.format(the_world.agent))
        print('patients: {}'.format(the_world.p_noun))
        print('nouns:{}'.format(the_world.agent + the_world.p_noun))
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
        print('{} carnivor hunting success'.format(the_world.hunt_success))
        print('{} herbivores left'.format((len(the_world.acting_herbivore_list))))
        for pair in sorted_t_p_pairs:
            print(pair)
        for pair in sorted_v_a_pairs:
            print(pair)
        #for pair in sorted_collapsed_pairs:
            #print(pair)
    return the_world


# syntagmatic rules for whether or not a noun can be the agent or patient of an event (verb)

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
    # there can be multiple humans in the world, in the current model, there is only one
    profiles = []  # a record of information with respect to individuals: individuals are keys, and values are the
    # experiment-relevant information regarding the individuals

    profile = {}  # currently a paradigmatic and a syntagmatic task are carried out,  pass the input to the
    # task, with respective to each individual
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
    nouns = agent + p_nouns
    pairs = the_world.pairs
    collapsed_pairs = the_world.collapsed_pairs
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
    kit['p_nouns'] = p_nouns # patient nouns
    kit['t_verbs'] = t_verbs # transitive verbs
    kit['nouns'] = nouns # all nouns
    kit['agent'] = agent # all agents
    kit['verbs'] = verbs # all verbs
    kit['v_a_pairs'] = v_a_pairs # occurred verb-agent pairs
    kit['t_p_pairs'] = t_p_pairs # occurred transitive-patient pairs
    kit['pairs'] = pairs # all occurred pairs\
    kit['collapsed_pairs'] = collapsed_pairs
    kit['flat_item'] = flat_item
    kit['the_world'] = the_world # world information
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