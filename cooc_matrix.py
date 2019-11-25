import numpy as np
from cytoolz import itertoolz

PAD = '*PAD*'
VERBOSE = False


def create_ww_matrix(vocab_list, vocab_index_dict, tokens):  # no function call overhead - twice as fast
    window_type = 'forward'
    window_size = 7
    window_weight = 'flat'

    # count
    num_vocab = len(vocab_list)
    count_matrix = np.zeros([num_vocab, num_vocab], int)

    print('\nCounting word-word co-occurrences in {}-word moving window'.format(window_size))

    for i in range(window_size):
        tokens.append(PAD)

    windows = itertoolz.sliding_window(window_size + 1, tokens)  # + 1 because window consists of t2s only
    for window in windows:
        print(window)
        if window[0] in vocab_index_dict:
            for i in range(window_size):
                if window[i+1] in vocab_index_dict:
                    if window_weight == "linear":
                        count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i+1]]] += window_size - dist
                    elif window_weight == "flat":
                        count_matrix[vocab_index_dict[window[0]], vocab_index_dict[window[i+1]]] += 1

    # window_type
    if window_type == 'forward':
        final_matrix = count_matrix
    elif window_type == 'backward':
        final_matrix = count_matrix.transpose()
    elif window_type == 'summed':
        final_matrix = count_matrix + count_matrix.transpose()
    elif window_type == 'concatenated':
        final_matrix = np.concatenate((count_matrix, count_matrix.transpose()), axis=1)
    else:
        raise AttributeError('Invalid arg to "window_type".')
    print('Shape of normalized matrix={}'.format(final_matrix.shape))
    return final_matrix


def main():
    vocab_list = ['the', 'dog', 'cat', 'mouse', 'chased', '.']
    vocab_index_dict = {'the': 0,
                        'dog': 1,
                        'cat': 2,
                        'mouse': 3,
                        'chased': 4,
                        '.': 5}
    corpus = "the dog chased the cat . the cat chased the mouse .".split()

    final_matrix = create_ww_matrix(vocab_list, vocab_index_dict, corpus)
    print(final_matrix)

main()