from Programs.World import config
import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

plot_size = 20

two_encode_types = True
thematic = True # whether or not the data is generated with separate thematic roles
num_run = 2

path = Path().cwd().parent.parent / 'Data' / 'direct_cos_non_thematic_spearman.csv'
model_path = Path().cwd().parent.parent / 'Data' / 'model_dict.csv'
mean_path = Path().cwd().parent.parent / 'Data' / 'mean_pd.csv'
rank_path = Path().cwd().parent.parent / 'Data' / 'g_s_relatedness_cos_log_thematic_spearman.csv'

save_path1 = str(Path().cwd().parent.parent / 'Data' / 's_performance_thematic_spearman')
save_path2 = str(Path().cwd().parent.parent / 'Data' / '3ways')
save_path3 = str(Path().cwd().parent.parent / 'Data' / '2way_violin_thematic_spearman')
save_pd_mean =  str(Path().cwd().parent.parent / 'Data' / 'mean_pd.csv')
save_path_rank_corr = str(Path().cwd().parent.parent / 'Data' / '2way_violin_rank_corr')

if two_encode_types:
    save_raw_pd = str(Path().cwd().parent.parent / 'Data' / 'raw_2by2.csv')
else:
    save_raw_pd = str(Path().cwd().parent.parent / 'Data' / 'raw_2by7.csv')

save_path_2by2 =  str(Path().cwd().parent.parent / 'Data' / '2by2.png')




# caculate mean performance from stored data

def get_performance(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_data = row[2:]
            if csv_reader.line_num == 1:
                models = row_data
            if csv_reader.line_num == 2:
                data_matrix = np.asmatrix(row_data, float)

            elif csv_reader.line_num > 2:
                current_matrix = np.asmatrix(row_data, float)
                data_matrix = np.concatenate((data_matrix, current_matrix), 0)
        #data_matrix = data_matrix[1:]
        data_matrix = np.asarray(data_matrix)
        n_row = data_matrix.shape[0]
        mean = data_matrix.mean(0)
        std_error = data_matrix.std(0)
    return mean, std_error, models, n_row

# reorganize the ranking(relatedness) matrix by category (human, carnivore, fruit) instead of items (rabbit, flower)
# across all experimental runs. Only the verbs shared by all runs are included.
def get_ranking(r_path):
    rank_df = pd.read_csv(r_path)

    shared_verb = set({})
    categories = set({})
    for i in range(num_run):
        run_df = rank_df[rank_df['world'] == i+1]
        run_verbs = set(run_df['verb'])
        run_categories = set(run_df['noun category'])
        if i == 0:
            shared_verb = shared_verb.union(run_verbs)
            categories = categories.union(run_categories)
        else:
            shared_verb = shared_verb.intersection(run_verbs)
            categories = categories.intersection(run_categories)
    shared_verb = list(shared_verb)
    categories = list(categories)
    shared_verb.sort()
    categories.sort()
    n_row = num_run * len(shared_verb) * len(categories)
    n_col = len(rank_df.columns) - 5
    run_size = len(shared_verb) * len(categories)
    print(run_size)
    rank_matrix = np.zeros((n_row,n_col))
    for i in range(num_run):
        for j in range(len(shared_verb)):
            verb = shared_verb[j]
            for k in range(len(categories)):
                id_row = i * len(shared_verb) * len(categories) + j * len(categories) + k
                category = categories[k]

                sub_df = rank_df[(rank_df['world']== i+1) & (rank_df['verb'] == verb) & (rank_df['noun category'] == category)]
                sub_df = sub_df[sub_df.columns[5:]]

                rank_matrix[id_row] = sub_df.to_numpy().mean(axis = 0)

    return rank_matrix, run_size

# caculate the correlations of each model across runs

def get_rank_correlatons(rank_matrix, num_run, run_size):
    n_row = int(num_run * (num_run - 1) / 2)
    n_col = rank_matrix.shape[1]

    corr_matrix = np.zeros((n_row,n_col))


    for i in range(num_run - 1):
        for j in range(i+1, num_run):
            id_row = int((2*num_run-i-1)*i/2 + j - i - 1)
            for k in range(n_col):
                start_i = i * run_size
                end_i = (i+1)*run_size
                start_j = j * run_size
                end_j = (j + 1) * run_size
                corr_matrix[id_row][k] =  np.corrcoef(rank_matrix[start_i:end_i,k],
                                                      rank_matrix[start_j:end_j,k])[0][1]


    mean = corr_matrix.mean(0)
    std_error = corr_matrix.std(0)


    model_means = mean[1:]
    model_stds = std_error[1:]

    model_mean = mean[1:].mean()
    model_std = mean[1:].std()

    print('target')
    print(mean[0],std_error[0])
    print('model')
    print(model_mean, model_std)

    return model_means, model_stds


# organize the data into a panda data frame

def get_dataframe(mean, std_error, dict_path):
    mean_pd = pd.read_csv(dict_path)
    mean_pd['mean'] = mean
    mean_pd['std_error'] = std_error
    mean_pd.to_csv(save_pd_mean)
    return mean_pd

# store raw data in the parameterized dataframe

def get_raw_dataframe(dict_path, raw_data_path):
    raw_pd = pd.read_csv(dict_path)
    add_pd = pd.read_csv(dict_path)

    if two_encode_types:
        raw_pd['encode'] = raw_pd['encode'].replace(['distance','cos','corr','r_distance','r_cos','r_corr'],'sim')
        add_pd['encode'] = raw_pd['encode'].replace(['distance', 'cos', 'corr', 'r_distance', 'r_cos', 'r_corr'], 'sim')


    with open(raw_data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_data = row[2:]
            if csv_reader.line_num == 2:
                data_matrix = np.asmatrix(row_data, float)
            elif csv_reader.line_num > 2:
                current_matrix = np.asmatrix(row_data, float)
                data_matrix = np.concatenate((data_matrix, current_matrix), 0)
        data_matrix = np.asarray(data_matrix)
        n_row = data_matrix.shape[0]
        n_col = data_matrix.shape[1]

    raw_pd['run'] = n_col*['R1']

    for i in range(n_row-1):
        run = 'R' + str(i+2)
        add_pd['run'] = n_col*[run]
        raw_pd = raw_pd.append(add_pd,ignore_index=True)

    flat_data = data_matrix.flatten()
    for i in range(len(flat_data)):
        if flat_data[i] > 0.9999:
            flat_data[i] = 0.9999

    raw_pd['performance'] = flat_data
    raw_pd.to_csv(save_raw_pd)



def plot_2by2():
    df = pd.read_csv(mean_path)
    cooc_space = df.loc[np.logical_and(df['encode'] == 'cooc', df['representation'] == 'space')]['mean']
    cooc_graph = df.loc[np.logical_and(df['encode'] == 'cooc', df['representation'] == 'graph')]['mean']
    sim_space = df.loc[np.logical_and(df['encode'] != 'cooc', df['representation'] == 'space')]['mean']
    sim_graph = df.loc[np.logical_and(df['encode'] != 'cooc', df['representation'] == 'graph')]['mean']

    cooc_mean = [round(cooc_space.mean(axis = 0),3), round(cooc_graph.mean(axis = 0),3)]
    cooc_std = [round(cooc_space.std(axis=0),3), round(cooc_graph.std(axis=0),3)]
    cooc_max = [round(cooc_space.max(axis=0),3), round(cooc_graph.max(axis=0),3)]

    sim_mean = [round(sim_space.mean(axis=0),3), round(sim_graph.mean(axis=0),3)]
    sim_std = [round(sim_space.std(axis=0),3), round(sim_graph.std(axis=0),3)]
    sim_max = [round(sim_space.max(axis=0),3), round(sim_graph.max(axis=0),3)]

    labels = ['Space','Graph']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, cooc_mean, width, label='Co-occurrence', yerr = None)
    rects2 = ax.bar(x + width / 2, sim_mean, width, label='Similarity', yerr = None)
    for i in x:
        ax.scatter([i - width/2, i + width/2], [cooc_max[i], sim_max[i]])
    rect_set = [rects1, rects2]
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Performance')
    ax.set_title('Data Structure by Encoding Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for rects in rect_set:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path_2by2)


# get parameters and build the parameter dictionary (for model variations, so that can check the parameter values of
# each model variation

def get_parameter_dict(dict_path):
    parameter_dict = {}
    with open(dict_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_data = row[1:]
            num_para = len(row_data)
            if csv_reader.line_num == 1:
                parameters = row_data
                for parameter in parameters:
                    parameter_dict[parameter] = []
            else:
                current_parameters = row_data
                for i in range(num_para):
                    if current_parameters[i] not in parameter_dict[parameters[i]]:
                        parameter_dict[parameters[i]].append(current_parameters[i])

    return parameters, parameter_dict


def get_hd_index(num, carry):
    index = []
    for digit in carry:
        divided = math.floor(num/digit)
        index.append(divided)
        num = num % digit
    index = tuple(index)
    return index



# store performance data in high dimensional matrix (not necessary)

def get_hd_matrix(parameter_dict, mean):
    matrix_shape = []
    current_index = []
    carry = []
    para_size = 1
    for parameter in parameter_dict:
        matrix_shape.append(len(parameter_dict[parameter]))
        current_index.append(0)
        para_size = para_size * len(parameter_dict[parameter])

    matrix_shape = tuple(matrix_shape)
    hd_matrix = np.zeros(matrix_shape)

    for num in matrix_shape:
        para_size = para_size/num
        carry.append(int(para_size))


    for i in range(len(mean)):
        hd_index = get_hd_index(i, carry)
        #print(hd_index)
        hd_matrix[hd_index] = mean[i]

    return hd_matrix


# sort the model variations by their mean performances, some model variations may have identical performances

def get_sorted(mean, std_error, models, n_row, rank_corr_mean, rank_corr_std):
    plot_dict = {}
    encode = ['distance','cos', 'corr','r_distance', 'r_cos', 'r_corr','cooc']
    representation = ['space', 'graph']
    best_dict = {}
    for i in range(len(models)):
        model = 'M' + str(i+1)
        model_encode = encode[math.ceil(((i+1) % 14) / 2) - 1]
        model_representation = representation[i%2]
        if model_encode not in best_dict:
            if model_representation == 'space':
                best_dict[model_encode] = [i+1, 0]
            else:
                best_dict[model_encode] = [0, i+1]
        else:
            if model_representation == 'space':
                if best_dict[model_encode][0] == 0:
                    best_dict[model_encode][0] = i+1
                else:
                    current_best_space = best_dict[model_encode][0]
                    if mean[i] > mean[current_best_space-1]:
                        best_dict[model_encode][0] = i+1
            else:
                if best_dict[model_encode][1] == 0:
                    best_dict[model_encode][1] = i + 1
                else:
                    current_best_graph = best_dict[model_encode][1]
                    if mean[i] > mean[current_best_graph - 1]:
                        best_dict[model_encode][1] = i + 1

        if mean[i] not in plot_dict:
            plot_dict[mean[i]] = [[model], std_error[i]]
        else:
            plot_dict[mean[i]][0].append(model)
    sorted_mean = sorted(plot_dict.keys(), reverse = True)
    sorted_se = []
    sorted_model = []
    sorted_model_dict = {}
    for key in sorted_mean:
        rep_model = plot_dict[key][0][0]
        sorted_model.append(rep_model)
        sorted_model_dict[rep_model] = plot_dict[key][0]
        sorted_se.append(plot_dict[key][1]/n_row)

    i = 1
    for rep_model in sorted_model_dict:
        model_num = int(rep_model[1:])
        model_encode = encode[math.ceil((model_num%14)/2)-1]
        print(i, rep_model, sorted_model_dict[rep_model], model_encode, sorted_mean[i-1],
              (rank_corr_mean[i-1], rank_corr_std[i-1]))
        i = i + 1
    print(best_dict)
    print()
    for encode in best_dict:

        print(encode, best_dict[encode], mean[best_dict[encode][0]-1],mean[best_dict[encode][1]-1])
    return sorted_model, sorted_mean, sorted_se


# make barplot of the model performances

def plot_performance(sorted_model, sorted_mean, sorted_se):
    num_model = len(sorted_mean)
    num_plot = math.ceil(num_model/plot_size)
    for i in range(num_plot):
        if (i+1)*plot_size > num_model:
            size = num_model - i*plot_size
        else:
            size = plot_size
        x_pos = np.arange(size)
        plot_mean = sorted_mean[i*plot_size:i*plot_size+size]
        plot_se = sorted_se[i*plot_size:i*plot_size+size]
        plot_models = sorted_model[i*plot_size:i*plot_size+size]
        colors = []
        for model in plot_models:
            num = int(model[1:])
            if num % 2 == 0:
                colors.append('blue')
            else:
                colors.append('red')
        colors = tuple(colors)
        fig, ax = plt.subplots()
        ax.bar(x_pos, plot_mean, yerr=plot_se, align='center', alpha=0.5, ecolor='black', color=colors)
        ax.set_ylabel('Avg Syntagmatic Correlation', size = 20)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_models, size = 15)
        label = []
        for y in plot_mean:
            label.append(str(round(y, 3)))
        for k in x_pos:
            #if k % 3 == 0:
                if plot_mean[k] > 0:
                    ax.text(x=x_pos[k] - 0.35, y=plot_mean[k] + 0.015, s=label[k], size=15)
                else:
                    ax.text(x=x_pos[k] - 0.35, y=plot_mean[k] - 0.03, s=label[k], size=15)

        ax.set_title(str(i+1))
        ax.yaxis.grid(True)
        fig.set_size_inches(20,7)
        plt.yticks(fontsize = 15)
        plt.tight_layout()
        plt.savefig(save_path1 + '/' + str(i+1) + '.png')


# collopse according to all different combination of parameters

def get_moment_index(num):
    moment_indices = {}
    for i in range(num+1):
        moment_indices[i] = []
        if i == 0:
            moment_indices[i].append([-1])
        else:
            last_indices = moment_indices[i-1]
            for index in last_indices:
                start = index[-1] + 1
                if start < num:
                    append_list = range(start,num)
                    for to_append in append_list:
                        new_index = index + [to_append]
                        if i == 1:
                            new_index = new_index[1:]
                        moment_indices[i].append(new_index)
    indice_list = moment_indices[0]
    for i in range(1,num+1):
        indice_list.extend(moment_indices[i])
        indice_list = indice_list[1:]
    return indice_list, moment_indices


def get_coordinate_space(shape):
    coordinates = {}
    for i in range(len(shape)):
        size = shape[i]
        coordinates[i] = []
        if i == 0:
            for j in range(size):
                coordinates[i].append([j])
        else:
            for coordinate in coordinates[i-1]:
                for j in range(size):
                    coordinates[i].append( coordinate + [j])
    coordinate_space = coordinates[len(shape)-1]

    return coordinate_space


def dim_collapse(hd_matrix, parameter_dict, parameters, interact_dim):
    num_para = len(parameters)
    dim_set = set(range(num_para))
    interact_set = {-1}
    for dim in interact_dim:
        interact_set = interact_set.union({parameters.index(dim)})
    reduced_dim = dim_set - interact_set
    reduced_dim = list(reduced_dim)
    reduced_dim.sort()
    reduced_dim = tuple(reduced_dim)
    dim_max = np.amax(hd_matrix, axis=reduced_dim)
    dim_max = np.asarray(dim_max)
    dim_mean = np.mean(hd_matrix, axis=reduced_dim)
    dim_mean = np.asarray(dim_mean)
    dim_err = np.std(hd_matrix, axis=reduced_dim)
    dim_err = np.asarray(dim_err)
    collapse_shape = dim_mean.shape
    coordinate_space = get_coordinate_space(collapse_shape)
    dim_err = dim_err/len(coordinate_space)
    plot_3way(dim_mean, dim_err, interact_dim, parameters, parameter_dict, 'mean')
    plot_3way(dim_max, np.zeros(collapse_shape), interact_dim, parameters, parameter_dict, 'max')


# given the 3 dimension matrix, plot bar graph to show three way interaction in the parameter space

def plot_3way(mean_matrix, error_matrix, chosen_para, parameters, parameter_dict, type):
    y_lim = 1.5 * np.amax(mean_matrix)
    chosen_id = []
    for variable in chosen_para:
        chosen_id.append(parameters.index(variable))
    chosen_id.sort()
    row_para = chosen_para[1]
    row_paras = parameter_dict[row_para]
    row_num = mean_matrix.shape[chosen_id.index(parameters.index(row_para))]
    col_para = chosen_para[2]
    col_paras = parameter_dict[col_para]
    col_num = mean_matrix.shape[chosen_id.index(parameters.index(col_para))]
    x_id = chosen_id.index(parameters.index(chosen_para[0]))
    row_id = chosen_id.index(parameters.index(chosen_para[1]))
    plot_x = np.arange(mean_matrix.shape[x_id])
    fig, ax = plt.subplots(row_num, col_num)
    for i in range(row_num):
        for j in range(col_num):
            if x_id == 0:
                if row_id == 1:
                    to_plot = mean_matrix[:,i,j]
                    err = error_matrix[:,i,j]
                else:
                    to_plot = mean_matrix[:,j,i]
                    err = error_matrix[:,j,i]
            elif x_id == 1:
                if row_id == 0:
                    to_plot = mean_matrix[i,:,j]
                    err = error_matrix[i,:,j]
                else:
                    to_plot = mean_matrix[j, :, i]
                    err = error_matrix[j, :, j]
            else:
                if row_id == 0:
                    to_plot = mean_matrix[i,j,:]
                    err = error_matrix[i,j,:]
                else:
                    to_plot = mean_matrix[j, i, :]
                    err = error_matrix[j, i, :]
            ax[i,j].bar(plot_x,to_plot, yerr = err)
            ax[i,j].set_ylim([-y_lim, y_lim])
            ax[i,j].set_xticks(plot_x)
            ax[i,j].set_xticklabels(tuple(parameter_dict[chosen_para[0]]))
            label = []
            for y in to_plot:
                label.append(str(round(y,3)))
            for k in plot_x:
                if to_plot[k] > 0:
                    ax[i,j].text(x=plot_x[k]-0.1, y=to_plot[k] + 0.1 * y_lim, s=label[k], size = 10)
                else:
                    ax[i, j].text(x=plot_x[k] -0.1, y=to_plot[k] - 0.2 * y_lim, s=label[k], size = 10)
            if j == 0:
                ax[i,j].set_ylabel(row_para + ":" + str(row_paras[i]))
            if i == 0:
                ax[i,j].set_title(col_para + ':' + str(col_paras[j]))
    fig.set_size_inches(15,10)
    fig.suptitle(type + ' in subspace ' + str(chosen_para))
    plt.savefig(save_path2 + '/' + type + '/' + chosen_para[0]+'_'+chosen_para[1]+'_'+chosen_para[2] + '.png')
    # plt.show()


# make violin plot to show distribution of 2X7 conditions

def plot_rep_violin(pandas_data, x, y, hue, save_path):
    plt.figure(figsize = (12.5,8))
    sns.violinplot(x = y, y = x, hue = hue, data = pandas_data, palette="Set2",
                              split=True, scale= 'count', inner="stick", scale_hue=False, bw=.1)
    if y == 'mean':
        plt.xlabel('X-Axis: Average Correlation', fontsize = 12)
        plt.legend(loc='upper left')
    else:
        plt.xlabel('X-Axis: Standard Error', fontsize = 15)
        plt.legend(loc='upper right')
    plt.ylabel(x, color = 'w')
    if x == 'encode':
        plt.yticks([0,1,2,3,4,5,6],['distance','cosine','correlation','reduced-distance','reduced-cosine',
                                    'reduced-correlation','co-occurrence'], fontsize = 12)

    plt.savefig(save_path + '/' + hue + '_' + x + '_' + y, bbox_inches='tight')


def plot_interaction(indices, parameter_dict, parameters, data_frame, n_way, save_path, x_name):
    if n_way == 3:
        for index in indices:
            chosen_para = []
            for i in index:
                chosen_para.append(parameters[i])
            dim_collapse(data_frame, parameter_dict, parameters, chosen_para)
    else:
        sns.set(style="whitegrid")
        for para in parameters:
            if para == 'encode':
                plot_rep_violin(data_frame, para, x_name, 'representation', save_path)

def best_model_relatedness():
    pass

def get_best_score():
    pass


def main():
    ranking_matrix, run_size = get_ranking(rank_path)
    rank_mean, rank_std = get_rank_correlatons(ranking_matrix, num_run, run_size)
    mean, std_error, models, n_row = get_performance(path)
    data_frame = get_dataframe(mean, std_error, model_path)
    sorted_model, sorted_mean, sorted_se = get_sorted(mean, std_error, models, n_row, rank_mean, rank_std)

    #plot_2by2()
    #plot_performance(sorted_model,sorted_mean,sorted_se)
    #parameters, parameter_dict = get_parameter_dict(model_path)
    #hd_matrix = get_hd_matrix(parameter_dict, mean)
    #indices3 = get_moment_index(len(parameters))[1][3]
    #plot_interaction(indices3, parameter_dict, parameters, data_frame, 2, save_path3, 'mean')
    plot_interaction(indices3, parameter_dict, parameters, data_frame, 2, save_path3, 'std_error')


    #rank_df = get_dataframe(rank_mean, rank_std, model_path)
    #plot_interaction(indices3, parameter_dict, parameters, rank_df, 2, save_path_rank_corr, 'mean')




#main()

get_raw_dataframe(model_path, path) # get the 2 by 2 or 2 by 7 dataframe










