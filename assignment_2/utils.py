import random
from collections import Counter

import torch
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt

# Paths
VOCAB_PATH = 'Data/vocab.txt'
WORD_VECTORS_PATH = 'Data/wordVectors.txt'


def read_vocab(vocab_path=VOCAB_PATH):
    with open(vocab_path, 'r') as f:
        lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]
    return lines



# Load the pre-trained word2vec
def load_word2vec(vocab_path=VOCAB_PATH, word_vectors_path=WORD_VECTORS_PATH):
    vecs = np.loadtxt(word_vectors_path)
    words = read_vocab(vocab_path)
    word2vec = {word: vecs[i] for i, word in enumerate(words)}
    # Add special tokens
    word2vec['<PAD_START>'] = np.zeros(50)
    word2vec['<PAD_END>'] = np.zeros(50)
    word2vec['<UNK>'] = np.ones(50)
    return word2vec


def cosine_similarity(u, v):
    if np.linalg.norm(u) == 0 or np.linalg.norm(v) == 0:
        return 0
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def most_similar(word, k, word2vec):
    if word not in word2vec:
        return []

    word_vec = word2vec[word]
    similarities = [(w, cosine_similarity(word_vec, vec)) for w, vec in word2vec.items() if w != word]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


# Read the data from the file and return the list of the sequences of words and tags
def read_data(filename, window_size=5):
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


def read_data_pre_suf(filename, window_size=5):
    sentences, prefixes, suffixes, tags = [], [], [], []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, prefixe, suffixe, sentence_tags = [], [], [], []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                prefixes.append(prefixe)
                suffixes.append(suffixe)
                tags.append(sentence_tags)
                sentence, prefixe, suffixe, sentence_tags = [], [], [], []
            else:
                word, tag = line.replace('\n', '').split()
                l = min(len(word), 3)
                sentence.append(word)
                prefixe.append(word[:l])
                suffixe.append(word[len(word) - l:])
                sentence_tags.append(tag)
    return sentences, prefixes, suffixes, tags


def read_test_data_pre_suf(filename, window_size=5):
    sentences, prefixes, suffixes, tags = [], [], [], []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, prefixe, suffixe = [], [], []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                prefixes.append(prefixe)
                suffixes.append(suffixe)
                tags.append(['<TEST>' for _ in range(len(sentence))])
                sentence, prefixe, suffixe = [], [], []
            else:
                word = line.replace('\n', '')
                l = min(len(word), 3)
                sentence.append(word)
                prefixe.append(word[:l])
                suffixe.append(word[len(word) - l:])

    return sentences, prefixes, suffixes, tags
def get_pre_suf_list(prefixes, suffixes):
    words = []
    [words.extend(s) for s in prefixes]
    [words.extend(s) for s in suffixes]
    return list(set(words))


def make_pre_suf(pre_suf):
    d = {w: i for i, w in enumerate(pre_suf)}
    d['<UNK>'] = len(d)
    return d


'''def read_test_data(filename, window_size=5):
    sentences = []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence, sentence_tags = [], []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                sentence_tags.append(['<TEST>' for _ in sentence])
                sentence = []
            else:
                word = line.replace('\n', '')
                sentence.append(word)

    return sentences, sentence_tags'''


# Read test data
def read_test_data(filename, window_size=5):
    sentences = []

    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        sentence = []
        for line in lines:
            if line == '\n':
                sentences.append(sentence)
                sentence = []
            else:
                word = line.replace('\n', '')
                sentence.append(word)

    return sentences


# making a words vocabulary and a tags vocabulary which are dictionaries that map words/tags to indices
def make_vocabs(words, tags):
    words = [w for sentence in words for w in sentence]
    words = list(set(words))
    words.append('<UNK>')
    words.append('<PAD_START>')
    words.append('<PAD_END>')
    tags = [t for tag in tags for t in tag]
    tags = sorted(set(tags))

    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for i, t in enumerate(tags)}

    return word2idx, idx2word, tag2idx, idx2tag


def make_vocabs_part4(words, tags):


    words.append('<UNK>')
    words.append('<PAD_START>')
    words.append('<PAD_END>')
    tags = [t for tag in tags for t in tag]
    tags = sorted(set(tags))

    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for i, t in enumerate(tags)}

    return word2idx, idx2word, tag2idx, idx2tag


def convert_words_to_window(words, tags=None, window_size=5):
    """
    @param words: a list of lists of words in the sentences
    @param tags: a list of lists of tags in the sentences
    @param window_size: the size of the window to use
    @return: a list of tuples that return the full vector of window_size*emb_size and the actual tag
    """
    vector_out = []
    tag_out = []
    padding = (window_size - 1) // 2

    if tags is None:
        for sentence in words:
            # Apply padding
            padded_sentence = ['<PAD_START>'] * padding + sentence + ['<PAD_END>'] * padding

            for i in range(padding, len(padded_sentence) - padding):
                # Extract the window
                window = padded_sentence[i - padding: i + padding + 1]
                vector_out.append(window)

        return vector_out

    for sentence, sentence_tag in zip(words, tags):
        # Apply padding
        padded_sentence = ['<PAD_START>'] * padding + sentence + ['<PAD_END>'] * padding

        for i in range(padding, len(padded_sentence) - padding):
            # Extract the window
            window = padded_sentence[i - padding: i + padding + 1]
            vector_out.append(window)
            tag_out.append(sentence_tag[i - padding])

    return vector_out, tag_out


def convert_window_to_window_idx(window, tag, word2idx, tag2idx):
    """
    @param window: a list of lists of words in the sentences
    @param tag: a list of lists of tags in the sentences
    @param word2idx: the dictionary that maps words to indices
    @param tag2idx: the dictionary that maps tags to indices
    @return: a list of tuples that return the full vector of window_size*emb_size and the actual tag as indices
    """
    vector_out = []
    tag_out = []
    if tag is None:
        for w in window:
            vector_out.append([word2idx[w] if w in word2idx else word2idx['<UNK>'] for w in w])
        return vector_out

    for w, t in zip(window, tag):
        vector_out.append([word2idx[w] if w in word2idx else word2idx['<UNK>'] for w in w])
        tag_out.append(tag2idx[t])
    return vector_out, tag_out


def convert_window_to_window_idx_presuf(window, tag, word2idx, pre_suf2idx, tag2idx):
    """
    @param window: a list of lists of words in the sentences
    @param tag: a list of lists of tags in the sentences
    @param word2idx: the dictionary that maps words to indices
    @param tag2idx: the dictionary that maps tags to indices
    @return: a list of tuples that return the full vector of window_size*emb_size and the actual tag as indices
    """
    vector_out = []
    tag_out = []
    for w, t in zip(window, tag):
        w = w[0]
        l = min(len(w), 3)
        vec_word_pre_suf = [word2idx[w] if w in word2idx else word2idx['<UNK>'],
             pre_suf2idx[w[:l]] if w in pre_suf2idx else pre_suf2idx['<UNK>'],
             pre_suf2idx[w[len(w) - l:]] if w in pre_suf2idx else pre_suf2idx['<UNK>']]

        vector_out.append(vec_word_pre_suf)
        tag_out.append(tag2idx[t])

    return vector_out, tag_out

def change_tag_of_rare_words(words, threshold=1, tag='<UNK>'):
    """
    Definition: words that appear less than threshold times are considered rare
    """
    word_count = Counter()
    for sentence in words:
        word_count.update(sentence)

    for i, sentence in enumerate(words):
        for j, word in enumerate(sentence):
            if word_count[word] <= threshold:
                words[i][j] = tag
    return words


# this function graph loss/acuracy over epochs
def make_graph(y, title, ylabel, filename, xlabel='Epochs'):
    x = range(1, len(y) + 1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    train_words, train_tags = read_data('Data/pos/train')
    dev_words, dev_tags = read_data('Data/pos/dev')
    # test_words,_ = read_test_data('./../Data/pos/test')

    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(train_words, train_tags)
    windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=5)
    windows_idx, window_tags_idx = convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx)

    for i in range(10):
        print(windows[i], window_tags[i])

    # later
    # word2vec = word2vec(words, vectors)
    # vectors = read_vectors('../Data/wordVectors.txt')