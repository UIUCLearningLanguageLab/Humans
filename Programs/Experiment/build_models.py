period = True
from Programs.Spatial_Models import cooc_graph_analysis, spacial_analysis, sim_space_analysis, sim_graph_analysis

########################################################################################################################
# In Humans, corpora generated are lists of sentences, to feed the linear models, need to first transform the sentences
# into word tokens
# list of sentence into list of words, for spatial models
########################################################################################################################

def corpus_transformation(linear_corpus, period_mark):
    corpus = []
    vocab_index_dict = {}
    vocab_list = []
    for sentence in linear_corpus:
        for word in sentence:
            corpus.append(word)
            if word not in vocab_index_dict:
                l = len(vocab_list)
                vocab_index_dict[word] = l
                vocab_list.append(word)
        if period_mark:
            corpus.append('.')
    if period_mark:
        vocab_list.append('.')
        vocab_index_dict['.'] = len(vocab_list)-1
    return corpus, vocab_list, vocab_index_dict


cooc_matrix = spacial_analysis.build_cooc_space()[1]
cooc_graph = cooc_graph_analysis.build_cooc_graph()[1]
sim_matrix = sim_space_analysis.build_sim_space()[1]
sim_graph = sim_graph_analysis.build_sim_graph()[1]









