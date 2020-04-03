import math
import numpy as np
from Programs.Experiment import generate_world, build_models, paradigmatic_task , syntagmatic_task, output

num_run = 1
s_task = True
p_task = True
models = ['cooc', 'sim', 'cooc_graph', 'sim_graph']
parameters = ['window_size', 'window_weight', 'window_type', 'normalization', 'reduction', 'sim_type']

parameter_dict = {'window_size': [1],
                   'window_weight': ['linear'],
                   'window_type': ['forward','backward','summed'],  # , 'backward', 'summed', 'concatenated'],
                   'normalization': ['ppmi', 'non'],
                   'reduction': ['non','svd'],
                   'sim_type': ['cos', 'distance', 'corr']}


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
    num_model = len(models) * len(model_para_list)
    corr_header = ['subject']
    corr_dict = {'subject':[]}
    ranking_header = ['subject','item']
    ranking_dict = {'subject':[],'item':[]}
    wb_header = ['subject']
    wb_dict = {'subject':[]}
    corr_data_matrix = np.zeros((num_run, num_model))
    wb_data_matrix = np.zeros((4*num_run, num_model))
    ranking_data_matrix = np.zeros((1, num_model))

    for i in range(num_run):
        # create a world and get world info
        the_world = generate_world.running_world()
        profiles, linear_corpus = generate_world.get_world_info(the_world)
        corpora.append(linear_corpus)
        profile = profiles[0]
        kit = profile['kit']
        p_nouns = kit['p_nouns']
        if p_task:
            for j in range(4):
                wb_dict['subject'].append(i+1)
        if s_task:
            t_verbs = kit['t_verbs']
            corr_dict['subject'].append(i+1)
            for verb in t_verbs:
                for noun in p_nouns:
                    phrase = verb + '_' + noun
                    ranking_dict['item'].append(phrase)
                    ranking_dict['subject'].append(i+1)
            standard_ranking = syntagmatic_task.get_standard_ranking(kit)
            current_ranking_matrix = standard_ranking.reshape(len(standard_ranking),1)

        # use world info to build 4 models according each model parameter
        for model_parameters in model_para_list:
            print(model_parameters)
            id_parameters = model_para_list.index(model_parameters)
            word_bag, vocab_list, vocab_index_dict = build_models.corpus_transformation(linear_corpus)
            kit['vocab_list'] = vocab_list
            kit['vocab_index_dict'] = vocab_index_dict
            cooc_matrix, sim_matrix = build_models.build_model(word_bag, vocab_list, vocab_index_dict, model_parameters)
            kit['cooc_matrix'] = cooc_matrix
            kit['sim_matrix'] = sim_matrix

            for model in models:
                id_model = models.index(model) + 4 * id_parameters
                if p_task:
                    within_between = paradigmatic_task.run_task(kit,model)[1]
                    for j in range(4):
                        wb_data_matrix[4*i+j][id_model] = within_between[j]

                if s_task:
                    model_corr, output_ranking = syntagmatic_task.run_task(kit,model)
                    print(model_corr)
                    corr_data_matrix[i][id_model] = model_corr
                    current_ranking_matrix = np.concatenate((current_ranking_matrix, output_ranking), 1)


        if s_task:
            current_ranking_matrix = current_ranking_matrix[0:,1:]
            ranking_data_matrix = np.concatenate((ranking_data_matrix,current_ranking_matrix),0)
    if s_task:
        ranking_data_matrix = ranking_data_matrix[1:]

    output.output_corpora(corpora, num_run)
    if s_task:
        output.output_exp(num_model, corr_header, corr_dict, 's_corr.csv', corr_data_matrix)
        output.output_exp(num_model, ranking_header, ranking_dict, 's_ranking.csv', ranking_data_matrix)
    if p_task:
        output.output_exp(num_model, wb_header, wb_dict, 'within_between.csv', wb_data_matrix)

run_experiment(num_run)



















