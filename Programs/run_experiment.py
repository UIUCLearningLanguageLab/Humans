import math
import numpy as np
import networkx as nx
from pathlib import Path
from matplotlib import pyplot as plt
from Programs.Experiment import generate_world, build_models, paradigmatic_task , syntagmatic_task, output

# Experiment type:
# test world only for world generation
# s_task for conducting syntagmatic tasks
# p_task for conducting paradigmatic tasks (not investigated in the current project)

num_run = 1
test_world = False
graphical_distance = False # whether add graphical distance in addition to spreading activation for SR in graphical models
encoding_thematic = True # whether or not nouns are separated by thematic roles (for encoding)
evaluation_thematic = True # whether or not the nouns are separated by thematic roles when forming target (from corpus)
spearman = True # whether correlate ranks, or spearman-correlate the raw scores
measure = 'cos' # using which measure to form target [cos or dist]
normalization = 'non' # use ppmi or row-log to normalize target_matrix before computing target relatedness
LON = True #
s_task = True
p_task = False


svd_path = str(Path().cwd().parent / 'Data' / 'reduction')


parameters = ['period','boundary','window_size', 'window_weight', 'window_type', 'normalization',
              'encode','representation']

rep_para = ['representation']

if LON:
    parameter_dict = { 'period':['no'],
                   'boundary': ['yes'],
                   'window_size': [1],
                   'window_weight': ['flat'],
                   'window_type': ['summed'],  # , 'backward', 'summed', 'concatenated'],
                   'normalization': ['log'],
                   'encode': ['cos'],
                   'representation': ['graph']
                   }
else:
    parameter_dict = { 'period':['yes','no'],
                   'boundary': ['yes','no'],
                   'window_size': [1,2,7],
                   'window_weight': ['flat','linear'],
                   'window_type': ['forward','backward','summed'],  # , 'backward', 'summed', 'concatenated'],
                   'normalization': ['log','ppmi','non'],
                   'encode': ['distance','cos','corr','r_distance','r_cos','r_corr','cooc'],
                   'representation': ['space','graph']
                   }



rep_num = 1
for rep in rep_para:
    rep_num = rep_num * len(parameter_dict[rep])

chosen_para = ['representation','window_type', 'normalization']

def get_data_space():
    matrix_shape = []
    for parameter in parameter_dict:
        matrix_shape.append(len(parameter_dict[parameter]))
    matrix_shape = tuple(matrix_shape)
    hd_matrix = np.zeros(matrix_shape)
    return hd_matrix


# plot svd variances, to decide how many dimensions to keep for the word vectors
def plot_svd(cooc_var_list, path, run):
    size = len(cooc_var_list[0])
    num_row = int(len(cooc_var_list)/2)
    sum_var = np.zeros((size,))
    x = np.arange(size)
    x2 = np.arange(size - 1)
    var_matrix = np.zeros((num_row, size))
    var_matrix2 = np.zeros((num_row,size-1))
    c = 0
    c2 = 0
    for cooc_var in cooc_var_list:
        row_sum = cooc_var.sum()
        cooc_var = cooc_var / row_sum
        if cooc_var.shape == sum_var.shape:
            var_matrix[c] = cooc_var
            c = c+1
        else:
            var_matrix2[c2] = cooc_var
            c2 = c2 + 1
    avg_var = var_matrix.mean(0)
    avg_var2 = var_matrix2.mean(0)
    se = var_matrix.std(0)/num_row
    se2 = var_matrix2.std(0)/num_row
    plt.figure(figsize=(20,5))
    plt.errorbar(x, avg_var,  yerr=se, uplims=True, lolims=True, label = 'with period')
    plt.errorbar(x2, avg_var2,  yerr=se2, uplims=True, lolims=True, label = 'no period')
    plt.suptitle('World' + str(run))
    plt.ylabel('Avg Model eigenvalue')
    plt.xticks(x, x+1)
    plt.savefig(path + '/' + 'run' + str(run) + '.png')
    plt.figure(figsize=(20,5))

def get_model_list(parameter_dict, parameters):  # iterate over paramenter_dict to get the list of model_parameter_combinations (in
    # form of dictionary) in the experiment.
    model_list = []
    num_model = 1
    count_dict = {}
    for parameter in parameters:
        num_model = num_model*len(parameter_dict[parameter])
        count_dict[parameter] = num_model

    for i in range(num_model):
        model_dict = {'Model':'M' + str(i+1)}
        for parameter in parameter_dict:
            parameters = parameter_dict[parameter]
            id = math.floor((i*count_dict[parameter]/num_model)%len(parameters))
            model_dict[parameter] = parameters[id]
        model_list.append(model_dict)

    return model_list

## show how if the corpus get the right

def run_experiment(num_run):
    corpora = []
    model_para_list = get_model_list(parameter_dict,parameters)

    output.output_model_dict(parameters, model_para_list)
    num_model = len(model_para_list)
    corr_header = ['world']
    corr_dict = {'world':[]}
    ranking_header = ['world','item','verb','noun', 'noun category']
    ranking_dict = {'world':[],'item':[],'verb':[],'noun':[],'noun category':[]}
    verb_header = ['world','verb']
    verb_dict = {'world':[], 'verb':[]}
    wb_header = ['world'] # withing_between for paradigmatic
    wb_dict = {'world':[]}
    wb_data_matrix = np.zeros((4*num_run, num_model))
    ranking_data_matrix = np.zeros((1, num_model+1))
    relate_data_matrix = np.zeros((1, num_model+1))
    corr_data_matrix = np.zeros((num_run, num_model + 1))
    direct_corr_matrix = np.zeros((num_run, num_model + 1))
    indirect_corr_matrix = np.zeros((num_run, num_model + 1))
    verb_corr_matrix = np.zeros((1, num_model + 1))

    ranking_data_matrix_g = np.zeros((1, num_model + 1))
    relate_data_matrix_g = np.zeros((1, num_model + 1))
    corr_data_matrix_g = np.zeros((num_run, num_model + 1))
    direct_corr_matrix_g = np.zeros((num_run, num_model + 1))
    indirect_corr_matrix_g = np.zeros((num_run, num_model + 1))
    verb_corr_matrix_g = np.zeros((1, num_model + 1))
    for i in range(num_run):
        print()
        print('world ' + str(i))
        print()
        hd_matrix = get_data_space()

        # create a world and get world info
        the_world = generate_world.running_world()
        if test_world:
            continue
        profiles, linear_corpus = generate_world.get_world_info(the_world)
        corpora.append(linear_corpus)
        profile = profiles[0]
        kit = profile['kit']
        noun_stems = kit['noun_stems']
        nouns = kit['nouns']
        cooc_var_list = []

        # p_task:
        for j in range(4):
            wb_dict['world'].append(i+1)

        # s_task:
        verbs = kit['verbs']
        corr_dict['world'].append(i+1)
        if evaluation_thematic:
            for verb in verbs:
                verb_a = verb + '_a'
                verb_p = verb + '_p'
                verb_dict['verb'].append(verb_a)
                verb_dict['verb'].append(verb_p)
                verb_dict['world'].append(i+1)
                verb_dict['world'].append(i + 1)

                for noun in nouns:
                    phrase = verb + '_' + noun
                    ranking_dict['verb'].append(verb)
                    ranking_dict['item'].append(phrase)
                    ranking_dict['noun'].append(noun)
                    if noun[-1] == 'a':
                        ranking_dict['noun category'].append(the_world.noun_tax[noun[:-2]]+'_a')
                    else:
                        ranking_dict['noun category'].append(the_world.noun_tax[noun[:-2]]+'_p')
                    ranking_dict['world'].append(i+1)

        else:
            for verb in verbs:
                verb_dict['verb'].append(verb)
                verb_dict['world'].append(i+1)

                for noun in noun_stems:
                    phrase = verb + '_' + noun
                    ranking_dict['item'].append(phrase)
                    ranking_dict['verb'].append(verb)
                    ranking_dict['noun'].append(noun)
                    ranking_dict['noun category'].append(the_world.noun_tax[noun])
                    ranking_dict['world'].append(i + 1)

        standard_ranking, standard_thematic, standard_trivialities, direct_dict = \
            syntagmatic_task.get_standard_ranking(kit, measure, evaluation_thematic, normalization)


        standards = [standard_ranking, standard_thematic, standard_trivialities]

        if len(standard_ranking.shape) == 2:
            (n,m) = standard_ranking.shape
        else:
            n = 1
            m = standard_ranking.shape[0]
        current_ranking_matrix = standard_ranking.reshape(n * m,1)
        current_relate_matrix = standard_thematic.reshape(n * m,1)
        current_ranking_matrix_g = standard_ranking.reshape(n * m, 1)
        current_relate_matrix_g = standard_thematic.reshape(n * m, 1)
        if evaluation_thematic:
            current_verb_matrix = np.ones((2*len(verbs),1))
            current_verb_matrix_g = np.ones((2 * len(verbs), 1))
        else:
            current_verb_matrix = np.ones((len(verbs),1))
            current_verb_matrix_g = np.ones((len(verbs), 1))

        # use world info to build models according each model parameter
        for model_parameters in model_para_list:
            #print(model_parameters)
            #print()
            parameter_index = []
            for dimension in parameters:
                parameter_index.append(parameter_dict[dimension].index(model_parameters[dimension]))

            # generate the matrix for analysis
            id_parameters = model_para_list.index(model_parameters)
            if id_parameters % rep_num == 0:
                period = model_parameters['period']
                reduction = model_parameters['encode']
                if period == 'no':
                    period = False
                else:
                    period = True
                boundary = model_parameters['boundary']
                if boundary == 'yes':
                    boundary = True
                else:
                    boundary = False
                word_bag, vocab_list, vocab_index_dict = build_models.corpus_transformation(linear_corpus, period,
                                                                                            boundary, encoding_thematic)
                #print(word_bag)
                #print(vocab_list)
                #print(len(vocab_list))
                #print(vocab_index_dict)
                kit['vocab_list'] = vocab_list
                kit['vocab_index_dict'] = vocab_index_dict
                cooc_matrix, sim_matrix = build_models.build_model(word_bag, vocab_list, vocab_index_dict,
                                                                model_parameters)
                if LON:
                    id_milk_p = vocab_index_dict['milk-p']
                    id_drinking = vocab_index_dict['drinking']
                    print(cooc_matrix[id_drinking][id_milk_p])
                    print(cooc_matrix[id_milk_p][id_drinking])
                    print(id_milk_p,id_drinking)
                    cooc_net = nx.from_numpy_matrix(cooc_matrix,create_using=nx.MultiDiGraph())
                    mapping = {v: k for k, v in vocab_index_dict.items()}
                    cooc_net = nx.relabel_nodes(cooc_net, mapping)
                    path = Path().cwd().parent / 'Data' / "cooc_edgelist.csv"
                    nx.write_edgelist(cooc_net, path , delimiter=',', data=['weight'])

                kit['sim_matrix'] = sim_matrix
                if reduction[0] != 'r':
                    kit['cooc_matrix'] = cooc_matrix
                else:
                    kit['cooc_matrix'] = cooc_matrix[0]
                    cooc_var_list.append(cooc_matrix[1])

            # run models
            rep = model_parameters['representation']
            encode = model_parameters['encode']
            window_type = model_parameters['window_type']

            if encode == 'cooc' and (window_type == 'boundary' or window_type == 'summed'):
                dg = True
            else:
                dg = False

            if p_task:
                within_between = paradigmatic_task.run_task(kit, encode, rep)[1]
                for j in range(4):
                    wb_data_matrix[4*i+j][id_parameters] = within_between[j]

            if s_task:

                kit['model_num'] = model_parameters['Model']
                model_corr, output_ranking, output_relate, verb_corrs, direct_corr, indirect_corr = \
                    syntagmatic_task.run_task(kit, encode, rep, dg, evaluation_thematic, encoding_thematic, spearman,
                                              standards, direct_dict)

                current_ranking_matrix = np.concatenate((current_ranking_matrix, output_ranking), 1)
                current_relate_matrix = np.concatenate((current_relate_matrix, output_relate), 1)
                current_verb_matrix = np.concatenate((current_verb_matrix, verb_corrs), 1)
                corr_data_matrix[i][id_parameters + 1] = model_corr
                direct_corr_matrix[i][id_parameters + 1] = direct_corr
                indirect_corr_matrix[i][id_parameters + 1] = indirect_corr

                if rep == 'graph' and graphical_distance:
                    model_corr_g, output_ranking_g, output_relate_g, verb_corrs_g, direct_corr_g, indirect_corr_g = \
                        syntagmatic_task.run_task(kit, encode, rep, dg, evaluation_thematic, encoding_thematic,
                                                  spearman, standards, direct_dict, g_distance=True)

                    current_ranking_matrix_g = np.concatenate((current_ranking_matrix_g, output_ranking_g), 1)
                    current_relate_matrix_g = np.concatenate((current_relate_matrix_g, output_relate_g), 1)
                    current_verb_matrix_g = np.concatenate((current_verb_matrix_g, verb_corrs_g), 1)
                    corr_data_matrix_g[i][id_parameters] = model_corr_g
                    direct_corr_matrix_g[i][id_parameters + 1] = direct_corr_g
                    indirect_corr_matrix_g[i][id_parameters + 1] = indirect_corr_g

                    current_ranking_matrix_g = np.concatenate((current_ranking_matrix_g, output_ranking), 1)
                    current_relate_matrix_g = np.concatenate((current_relate_matrix_g, output_relate), 1)
                    current_verb_matrix_g = np.concatenate((current_verb_matrix_g, verb_corrs), 1)
                    corr_data_matrix_g[i][id_parameters + 1] = model_corr
                    direct_corr_matrix_g[i][id_parameters + 1] = direct_corr
                    indirect_corr_matrix_g[i][id_parameters + 1] = indirect_corr



                #print(model_corr_dict)
                #print()


            if id_parameters % 300 == 0:
                print(str(id_parameters) + ' models run')

        # show reduction var
        # plot_svd(cooc_var_list, svd_path, i)

        if s_task:
            #current_ranking_matrix = current_ranking_matrix[0:,1:]
            ranking_data_matrix = np.concatenate((ranking_data_matrix,current_ranking_matrix),0)
            #current_relate_matrix = current_relate_matrix[0:, 1:]
            relate_data_matrix = np.concatenate((relate_data_matrix, current_relate_matrix),0)
            verb_corr_matrix = np.concatenate((verb_corr_matrix, current_verb_matrix),0)
            if graphical_distance:
                ranking_data_matrix_g = np.concatenate((ranking_data_matrix_g, current_ranking_matrix_g), 0)
                # current_relate_matrix_g = current_relate_matrix_g[0:, 1:]
                relate_data_matrix_g = np.concatenate((relate_data_matrix_g, current_relate_matrix_g), 0)
                verb_corr_matrix_g = np.concatenate((verb_corr_matrix_g, current_verb_matrix_g), 0)

    if s_task:
        ranking_data_matrix = ranking_data_matrix[1:]
        relate_data_matrix = relate_data_matrix[1:]
        verb_corr_matrix = verb_corr_matrix[1:]
        if graphical_distance:
            ranking_data_matrix_g = ranking_data_matrix_g[1:]
            relate_data_matrix_g = relate_data_matrix_g[1:]
            verb_corr_matrix_g = verb_corr_matrix_g[1:]
    if not test_world:
        output.output_corpora(corpora, num_run)
    if s_task:
        file_name = measure + '_' + normalization + "_thematic" # 'thematic' if nouns separated by thematic roles, otherwiae 'collapsed'
        file_name = file_name + '_spearman' # 'spearman' if spearmon-correlation for raw relatedness; 'pearson' if
                                            # correlating ranks
        #file_name = file_name + '_test'

        performance_file_name = file_name + '.csv'
        output.output_exp(num_model, corr_header, corr_dict, performance_file_name, corr_data_matrix)
        output.output_exp(num_model, corr_header, corr_dict, 'direct_' + performance_file_name, direct_corr_matrix)
        output.output_exp(num_model, corr_header, corr_dict, 'indrect_' + performance_file_name, indirect_corr_matrix)

        if graphical_distance:
            g_performance_file_name = 'g_' + performance_file_name
            output.output_exp(num_model, corr_header, corr_dict, g_performance_file_name, corr_data_matrix_g)
            output.output_exp(num_model, corr_header, corr_dict, 'direct_' + g_performance_file_name, direct_corr_matrix_g)
            output.output_exp(num_model, corr_header, corr_dict, 'indrect_' + g_performance_file_name,
                              indirect_corr_matrix_g)

        verb_corr_name = 'verb_corr_' + file_name + '.csv'
        ranking_name = 's_ranking_' + file_name + '.csv'
        relatedness_name = 's_relatedness_' + file_name + '.csv'
        output.output_exp(num_model, verb_header, verb_dict, verb_corr_name,
                          verb_corr_matrix)
        output.output_exp(num_model, ranking_header, ranking_dict,  ranking_name, ranking_data_matrix)
        output.output_exp(num_model, ranking_header, ranking_dict,  relatedness_name, relate_data_matrix)
        if graphical_distance:
            g_verb_corr_name = 'g_' + verb_corr_name
            g_ranking_name = 'g_' + ranking_name
            g_relatedness_name = 'g_' + relatedness_name
            output.output_exp(num_model, verb_header, verb_dict, g_verb_corr_name,
                              verb_corr_matrix_g)
            output.output_exp(num_model, ranking_header, ranking_dict, g_ranking_name, ranking_data_matrix_g)
            output.output_exp(num_model, ranking_header, ranking_dict, g_relatedness_name, relate_data_matrix_g)

    if p_task:
        output.output_exp(num_model, wb_header, wb_dict, 'within_between.csv', wb_data_matrix)


run_experiment(num_run)



















