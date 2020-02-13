period = True


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