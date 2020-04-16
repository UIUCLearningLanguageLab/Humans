import csv
from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np

path = Path().cwd().parent.parent / 'Data' / 's_corr.csv'

save_path = str(Path().cwd().parent.parent / 'Data' / 's_performance')

plot_size = 10

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

plot_dict = {}


for i in range(len(models)):
    if i % 4 == 0:
        rep = 'C'
    elif i % 4 == 1:
        rep = 'S'
    elif i % 4 == 2:
        rep = 'CG'
    else:
        rep = 'SG'
    model = rep + str(math.floor(i/4)+1)
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

    fig, ax = plt.subplots()
    ax.bar(x_pos, plot_mean, yerr=plot_se, align='center', alpha=0.5, ecolor='black')
    ax.set_ylabel('Avg Syntagmatic Correlation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_models)
    ax.set_title(str(i+1))
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(save_path + '/' + str(i+1)+'.png')

for rep_model in sorted_model_dict:
    print(rep_model, sorted_model_dict[rep_model])










