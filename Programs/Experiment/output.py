import csv
from pathlib import Path


def output_exp(num_model, init_header, filename, data_matrix):
    path = Path().cwd().parent.parent / 'Data' / filename
    with path.open('w') as csvfile:  # save the standard ranking and model rankings in a csv file.
        fieldnames = init_header
        for i in range(num_model):
            model = 'M' + str(i)
            fieldnames.append(model)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_row = data_matrix.shape[0]
        num_column = len(fieldnames)
        for i in range(num_row):
            row = {}
            for j in range(num_column):
                row[fieldnames] = data_matrix[i][j]
            writer.writerow(row)


def output_model_dict(parameters, model_dict): # model_dict is a list of dictionaries, where each dictionary has
    # model label and parameters as keys
    path = Path().cwd().parent.parent / 'Data' / 'model_dict.csv'
    with path.open('w') as csvfile:  # save the standard ranking and model rankings in a csv file.
        fieldnames = ['Models'].append(parameters)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        num_row = len(model_dict)
        for i in range(num_row):
            writer.writerow(model_dict[i])
        

def output_corpora(corpora, num_run):
    path = Path().cwd().parent.parent / 'Data' / 'corpora.csv'
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

