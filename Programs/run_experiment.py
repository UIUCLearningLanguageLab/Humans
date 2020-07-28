import math
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from Programs.Experiment import generate_world, build_models, paradigmatic_task , syntagmatic_task, output

num_run = 1
test_world = True
s_task = False
p_task = False

svd_path = str(Path().cwd().parent / 'Data' / 'reduction')


parameters = ['period','boundary','window_size', 'window_weight', 'window_type', 'normalization',
              'encode','representation']

rep_para = ['representation']

parameter_dict = { 'period':['yes','no'],
                   'boundary': ['yes','no'],
                   'window_size': [1,2,7],
                   'window_weight': ['flat','linear'],
                   'window_type': ['forward','backward','summed'],  # , 'backward', 'summed', 'concatenated'],
                   'normalization': ['log','ppmi','non'],
                   'encode': ['cooc','cos','distance','corr','r_cos','r_distance','r_corr'],
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


def run_experiment(num_run):
    corpora = []
    model_para_list = get_model_list(parameter_dict,parameters)

    output.output_model_dict(parameters, model_para_list)
    num_model = len(model_para_list)
    corr_header = ['world']
    corr_dict = {'world':[]}
    ranking_header = ['world','item']
    ranking_dict = {'world':[],'item':[]}
    wb_header = ['world'] # withing_between for paradigmatic
    wb_dict = {'world':[]}
    corr_data_matrices = {}
    wb_data_matrix = np.zeros((4*num_run, num_model))
    ranking_data_matrix = np.zeros((1, num_model))
    relate_data_matrix = np.zeros((1, num_model))

    for i in range(num_run):
        print()
        print('world ' + str(i))
        print()
        hd_matrix = get_data_space()
        # create a world and get world info
        the_world = generate_world.running_world()
        if test_world:
            break
        profiles, linear_corpus = generate_world.get_world_info(the_world)
        corpora.append(linear_corpus)
        profile = profiles[0]
        kit = profile['kit']
        noun_stems = kit['noun_stems']
        cooc_var_list = []
        # p_task:
        for j in range(4):
            wb_dict['world'].append(i+1)
        # s_task:
        verbs = kit['verbs']
        corr_dict['world'].append(i+1)
        thematic_roles = ['a','p']
        for verb in verbs:
            for role in thematic_roles:
                for noun in noun_stems:
                    phrase = verb + '_' + noun + '_' + role
                    ranking_dict['item'].append(phrase)
                    ranking_dict['world'].append(i+1)
        standard_ranking = syntagmatic_task.get_standard_ranking(kit,'combine','cos')[0]
        current_ranking_matrix = standard_ranking.reshape(len(standard_ranking),1)
        current_relate_matrix = standard_ranking.reshape(len(standard_ranking),1)

        # use world info to build models according each model parameter
        for model_parameters in model_para_list:
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
                word_bag, vocab_list, vocab_index_dict = build_models.corpus_transformation(linear_corpus, period, boundary)
                #print(word_bag)
                #print(vocab_list)
                kit['vocab_list'] = vocab_list
                kit['vocab_index_dict'] = vocab_index_dict
                cooc_matrix, sim_matrix = build_models.build_model(word_bag, vocab_list, vocab_index_dict,
                                                                model_parameters)
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
                print(model_parameters)
                kit['model_num'] = model_parameters['Model']
                model_corr_dict, output_ranking, output_relate = syntagmatic_task.run_task(kit, encode, rep, dg)
                print(model_corr_dict)
                print()

                current_ranking_matrix = np.concatenate((current_ranking_matrix, output_ranking), 1)
                current_relate_matrix = np.concatenate((current_relate_matrix, output_relate), 1)
                if len(corr_data_matrices) == 0:
                    for standard in model_corr_dict:
                        corr_data_matrices[standard] = np.zeros((num_run, num_model))
                else:
                    for standard in model_corr_dict:
                        corr_data_matrix = corr_data_matrices[standard]
                        corr_data_matrix[i][id_parameters] = model_corr_dict[standard]

            if id_parameters % 300 == 0:
                print(str(id_parameters) + ' models run')

        # show reduction var
        # plot_svd(cooc_var_list, svd_path, i)

        if s_task:
            current_ranking_matrix = current_ranking_matrix[0:,1:]
            ranking_data_matrix = np.concatenate((ranking_data_matrix,current_ranking_matrix),0)
            current_relate_matrix = current_relate_matrix[0:, 1:]
            relate_data_matrix = np.concatenate((relate_data_matrix, current_relate_matrix),0)
    if s_task:
        ranking_data_matrix = ranking_data_matrix[1:]
        relate_data_matrix = relate_data_matrix[1:]
    if not test_world:
        output.output_corpora(corpora, num_run)
    if s_task:
        for standard in corr_data_matrices:
            corr_data_matrix = corr_data_matrices[standard]
            file_name = standard + ".csv"
            #output.output_exp(num_model, corr_header, corr_dict, file_name, corr_data_matrix)
        #output.output_exp(num_model, ranking_header, ranking_dict,  's_ranking.csv', ranking_data_matrix)
        #output.output_exp(num_model, ranking_header, ranking_dict,  's_relateness.csv', relate_data_matrix)
    if p_task:
        output.output_exp(num_model, wb_header, wb_dict, 'within_between.csv', wb_data_matrix)


run_experiment(num_run)



















