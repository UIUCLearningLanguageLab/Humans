period = True
from Programs.Spatial_Models import cooc_matrix, spacial_analysis

########################################################################################################################
# In Humans, corpora generated are lists of sentences, to feed the linear models, need to first transform the sentences
# into word tokens
# list of sentence into list of words, for spatial models
########################################################################################################################


def corpus_transformation(linear_corpus, period_mark = False):
    word_bag = []
    vocab_index_dict = {}
    vocab_list = []
    for sentence in linear_corpus:
        for word in sentence:
            word_bag.append(word)
            if word not in vocab_index_dict:
                l = len(vocab_list)
                vocab_index_dict[word] = l
                vocab_list.append(word)
        if period_mark:
            word_bag.append('.')
    if period_mark:
        vocab_list.append('.')
        vocab_index_dict['.'] = len(vocab_list)-1
    return word_bag, vocab_list, vocab_index_dict


def build_model(word_bag, vocab_list, vocab_index_dict, model_parameters):
    encoding = {}
    encoding['window_type'] = model_parameters['window_type']
    encoding['window_size'] = model_parameters['window_size']
    encoding['window_weight'] = model_parameters['window_weight']
    normalization = model_parameters['normalization']
    reduction = model_parameters['reduction']
    sim_type = model_parameters['sim_type']
    grand_matrix = cooc_matrix.get_cooc_matrix(vocab_list, vocab_index_dict, word_bag, encoding, normalization,
                                                   reduction)
    sim_matrix = spacial_analysis.get_sr_matrix(grand_matrix, vocab_list, vocab_list, vocab_index_dict, sim_type)
    return grand_matrix, sim_matrix











