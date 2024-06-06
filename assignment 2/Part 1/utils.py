import numpy as np
import torch
import itertools
from itertools import chain
from tqdm import tqdm

# Paths
vocab_path = 'Data/vocab.txt'
word_vectors_path = 'Data/wordVectors.txt'



# Return the list of words
def read_vocab(path='Data/vocab.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]
    return lines



# Return the word vectors
def read_vectors(path='Data/wordVectors.txt'):
    with open(path, 'r') as f:
        lines = f.readlines()

    lines = [[float(v) for v in l.replace('\n', '').split()] for l in lines]

    return torch.tensor(lines)

def word2vec(words, vectors):
    out = {w: v for w, v in zip(chain.from_iterable(words), vectors)}
    out['<PAD>'] = torch.zeros(50)
    out['<UNK>'] = torch.ones(50)
    return out
#TEST GIT COOL


# Read the data from the file and return the list of the sequences of words and tags
def read_data(filename, start_token='<s>', end_token='<e>'):
    sentences, tags = [], []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, sentence_tags = [], []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                tags.append(sentence_tags)
                sentence, sentence_tags = [], []
            else:
                word, tag = line.replace('\n', '').split()
                sentence.append(word)
                sentence_tags.append(tag)
    return sentences, tags


# Read test data
def read_test_data(filename, start_token='<s>', end_token='<e>'):
    sentences, full_sentences = [], []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, full_sentence = [], []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                full_sentences.append(full_sentence)
                sentence, full_sentence = [], []
            else:
                word = line.replace('\n', '')
                sentence.append(word)
                full_sentence.append(word)


# making a words vocabulary and a tags vocabulary which are dictionaries that map words/tags to indices
def make_vocabs(words, tags):
    words = [w for sent in words for w in sent]
    words = sorted(set(words))
    words.append('<UNK>')
    tags = [t for tag in tags for t in tag]
    tags = sorted(set(tags))

    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for i, t in enumerate(tags)}

    return word2idx, idx2word, tag2idx, idx2tag

#TODO
def convert_words_to_window(words, tags,word2vec,unknown_vec,padding_vec,window_size=5):
    """

    @param words:
    @param tags:
    @param window_size:

    @return: a list of tuples that return the full vector of window_size*emb_size and the actual tag
    """
    vector_out = []
    tag_out = []
    padding = (window_size - 1) // 2
    for sentence in tqdm(words):
        sentence = ['<PAD>']*padding + sentence + ['<PAD>']*padding
        for i, tag in zip(range(2, len(sentence) - 2), tags):
            vecs = []

            for window in range(-2,3):

                #print(word2vec[sentence[i + window]])
                w = sentence[i + window]
                if w in word2vec:
                    v = word2vec[sentence[i +window]]
                else:
                    v = word2vec['<UNK>']
                vecs.append(v)
            vector_out.append(torch.cat(vecs))
            tag_out.append(tag)

    return vector_out, tag_out






if __name__ == '__main__':

    padding_vec = torch.zeros(50)
    unknown_vec = torch.ones(50)

    words, tags = read_data('../Data/ner/train')
    vectors = read_vectors('../Data/wordVectors.txt')

    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(words, tags)
    word2vec = word2vec(words, vectors)
    vectors_ds, tag_ds = convert_words_to_window(words, tags, word2vec,unknown_vec,padding_vec)
    #print(word2idx)
    #print(idx2word)
    #print(tag2idx)
    #print(idx2tag)
    print(vectors_ds[0], tag_ds[0])
