import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np




def output_exp(num_model, init_header, init_dict, filename, data_matrix):
    print(init_header)
    print(init_dict)
    path = Path().cwd().parent / 'Data' / filename
    with path.open('w') as csvfile:  # save the standard ranking and model rankings in a csv file.
        fieldnames = []
        for head in init_header:
            fieldnames.append(head)
        init_len = len(init_header)
        for i in range(num_model+1):
            model = 'M' + str(i)
            fieldnames.append(model)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_row = data_matrix.shape[0]
        num_column = data_matrix.shape[1]
        for i in range(num_row):
            row = {}
            for field in init_header:
                row[field] = init_dict[field][i]
            for j in range(num_column):
                row[fieldnames[j+init_len]] = data_matrix[i][j]
            writer.writerow(row)


def output_model_dict(parameters, model_list): # model_dict is a list of dictionaries, where each dictionary has
    # model label and parameters as keys
    path = Path().cwd().parent / 'Data' / 'model_dict.csv'
    with path.open('w') as csvfile:  # save the standard ranking and model rankings in a csv file.
        fieldnames = ['Model'] + parameters
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_row = len(model_list)
        for i in range(num_row):
            writer.writerow(model_list[i])
        

def output_corpora(corpora, num_run):
    path = Path().cwd().parent / 'Data' / 'corpora.csv'
    with path.open('w') as csvfile: # save the standard ranking and model rankings in a csv file.
        fieldnames = ['Run','Sentence','Token', 'Word']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_run):
            row = {'Run': i}
            corpus = corpora[i]
            num_sent = len(corpus)
            for j in range(num_sent):
                row['Sentence'] = j
                sentence = corpus[j]
                num_word = len(sentence)
                for k in range(num_word):
                    row['Token'] = k
                    word = sentence[k]
                    row['Word'] = word
                    writer.writerow(row)


def corr_plot(hd_matrix, chosen_para, parameters, parameter_dict):
    # given the 3 dimension matrix, plot bar graph to show three way interaction in the parameter space
    collapse = list(range(len(parameter_dict)+1))
    chosen_id = []
    for variable in chosen_para:
        chosen_id.append(parameters.index(variable))
        collapse.remove(parameters.index(variable))
    chosen_id.sort()
    collapse = tuple(collapse)
    data_matrix = hd_matrix.mean(collapse)
    print(data_matrix.shape)
    row_para = chosen_para[1]
    row_paras = parameter_dict[row_para]
    row_num = data_matrix.shape[chosen_id.index(parameters.index(row_para))]
    col_para = chosen_para[2]
    col_paras = parameter_dict[col_para]
    col_num = data_matrix.shape[chosen_id.index(parameters.index(col_para))]
    x_id = chosen_id.index(parameters.index(chosen_para[0]))
    plot_x = np.arange(data_matrix.shape[x_id])
    fig, ax = plt.subplots(row_num, col_num)
    for i in range(row_num):
        for j in range(col_num):
            if x_id == 0:
                to_plot = data_matrix[:,i,j]
            elif x_id == 1:
                to_plot = data_matrix[i,:,j]
            else:
                to_plot = data_matrix[i,j,:]
            ax[i,j].bar(plot_x,to_plot)
            ax[i,j].set_ylim([-1, 1])
            ax[i,j].set_xticklabels(tuple(parameter_dict[chosen_para[0]]))
            if j == 0:
                ax[i,j].set_ylabel(row_para + ":" + str(row_paras[i]))
            if i == 0:
                ax[i,j].set_title(col_para + ':' + str(col_paras[j]))
    plt.savefig(chosen_para[0]+'_'+chosen_para[1]+'_'+chosen_para[2])
    plt.show()