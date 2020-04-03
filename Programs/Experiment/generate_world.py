from Programs.World import world, config

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


def get_world_info(the_world):
    # there could be multiple humans in the world, in the current model, there is only one
    profiles = []  # a record of information with respect to individuals: individuals are keys, and values are the
    # experiment-relevant information regarding the individuals
    for human in the_world.human_list:
        profile = {}  # currently a paradigmatic and a syntagmatic task are carried out, need to pass the input to the
        # task.
        output_info = {}
        kit = {} # a kit is the package of data in order to carrying out the tasks: including the
        # corpus, the verbs and nouns, and the verb-noun pairs and so on

        ###############################################################################################################
        # followings are preparation for the s_task and p_task in linear models, get the s_kit, p_kit for the tasks.
        ###############################################################################################################

        linear_corpus = human.linear_corpus
        p_nouns = human.p_noun
        t_verbs = human.t_verb
        pairs = human.t_p_pairs
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
        kit['pairs'] = pairs
        kit['flat_item'] = flat_item
        kit['the_world'] = the_world

        ###############################################################################################################
        # followings are preparation for the s_task and p_task in syntactic models, get the s_kit, p_kit for the tasks.
        ###############################################################################################################
        Steve = human.get_activated_words()[1]
        corpus = human.corpus
        num_sentence = len(corpus)
        #flat_standard = standard_ranking.flatten().reshape(rank_size,1)

        #print('standard')
        #print(standard_ranking)
        #recording_matrix = np.zeros((2 * len(window_sizes) + 3, len(window_weights) * len(window_types)))
        #data_matrix = flat_standard
        profile['kit'] = kit
        profile['output_info'] = output_info
        profiles.append(profile)
    return profiles, linear_corpus