import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

path = Path().cwd().parent.parent / 'Data' / 'cos_separate.csv'
model_path = Path().cwd().parent.parent / 'Data' / 'model_dict.csv'

save_path1 = str(Path().cwd().parent.parent / 'Data' / 's_performance')
save_path2 = str(Path().cwd().parent.parent / 'Data' / '3ways')
save_path3 = str(Path().cwd().parent.parent / 'Data' / '2way_violin')

plot_size = 33


# caculate mean performance from stored data

def get_performance(path):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            row_data = row[1:]
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

# organize the data into a panda data frame

def get_dataframe(mean, std_error, dict_path):
    mean_pd = pd.read_csv(dict_path)
    mean_pd['mean'] = mean
    mean_pd['std_error'] = std_error
    return mean_pd

# get parameters

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



# store performance data in high dimensional matrix

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


def get_sorted(mean, std_error, models, n_row):
    plot_dict = {}
    encode = ['cooc', 'cos', 'distance', 'corr', 'r_cos', 'r_distance', 'r_corr']
    for i in range(len(models)):
        model = 'M' + str(i+1)
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
    best_dict = {}
    for rep_model in sorted_model_dict:
        model_num = int(rep_model[1:])
        model_encode = encode[math.ceil((model_num%14)/2)-1]
        if model_encode not in best_dict:
            best_dict[model_encode] = [model_num]
        elif len(best_dict[model_encode]) < 2:
            exist_num = best_dict[model_encode][0]
            if model_num % 2 != exist_num % 2:
                best_dict[model_encode].append(model_num)
        print(i, rep_model, sorted_model_dict[rep_model], model_encode)
        i = i + 1
    print()
    for encode in best_dict:
        print(encode, best_dict[encode], mean[best_dict[encode][0]-1],mean[best_dict[encode][1]-1])
    return sorted_model, sorted_mean, sorted_se


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
        ax.set_ylabel('Avg Syntagmatic Correlation')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_models)
        label = []
        for y in plot_mean:
            label.append(str(round(y, 3)))
        for k in x_pos:
            if plot_mean[k] > 0:
                ax.text(x=x_pos[k] - 0.4, y=plot_mean[k] + 0.02, s=label[k], size=10)
            else:
                ax.text(x=x_pos[k] - 0.4, y=plot_mean[k] - 0.03, s=label[k], size=10)

        ax.set_title(str(i+1))
        ax.yaxis.grid(True)
        fig.set_size_inches(20,5)
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


def plot_3way(mean_matrix, error_matrix, chosen_para, parameters, parameter_dict, type):
    y_lim = 1.5 * np.amax(mean_matrix)
    # given the 3 dimension matrix, plot bar graph to show three way interaction in the parameter space
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

def plot_rep_violin(pandas_data, x, y, hue):
    plt.figure(figsize = (12.5,5))
    sns.violinplot(x = x, y = y, hue = hue, data = pandas_data, palette="Set2",
                              split=True, scale="count", inner="stick", scale_hue=False)
    plt.ylabel('Correlation score')
    plt.title('interaction between ' + hue + ' and ' + x)
    plt.savefig(save_path3 + '/' + hue + '_' + x )


def plot_interaction(indices, parameter_dict, parameters, data_frame, n_way):
    if n_way == 3:
        for index in indices:
            chosen_para = []
            for i in index:
                chosen_para.append(parameters[i])
            dim_collapse(data_frame, parameter_dict, parameters, chosen_para)
    else:
        sns.set(style="whitegrid")
        for para in parameters:
            if para != 'representation':
                plot_rep_violin(data_frame, para, 'mean', 'representation')

def best_model_relatedness():
    pass

def get_best_score():
    pass


def main():
    mean, std_error, models, n_row = get_performance(path)
    data_frame = get_dataframe(mean, std_error, model_path)
    sorted_model, sorted_mean, sorted_se = get_sorted(mean, std_error, models, n_row)
    #plot_performance(sorted_model,sorted_mean,sorted_se)
    parameters, parameter_dict = get_parameter_dict(model_path)
    #hd_matrix = get_hd_matrix(parameter_dict, mean)
    indices3 = get_moment_index(len(parameters))[1][3]
    plot_interaction(indices3, parameter_dict, parameters, data_frame, 2)

main()










