import csv
from pathlib import Path


def output_exp(num_model, init_header, init_dict, filename, data_matrix):
    print(init_header)
    print(init_dict)
    path = Path().cwd().parent / 'Data' / filename
    with path.open('w') as csvfile:  # save the standard ranking and model rankings in a csv file.
        fieldnames = []
        for head in init_header:
            fieldnames.append(head)
        init_len = len(init_header)
        for i in range(num_model):
            model = 'M' + str(i+1)
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