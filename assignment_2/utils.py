import torch
import numpy as np
from itertools import chain
from matplotlib import pyplot as plt

# Paths
VOCAB_PATH = '../Data/vocab.txt'
WORD_VECTORS_PATH = '../Data/wordVectors.txt'


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


# Read test data
def read_test_data(filename, window_size=5):
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
    words.append('<PAD_START>')
    words.append('<PAD_END>')
    tags = [t for tag in tags for t in tag]
    tags = sorted(set(tags))

    word2idx = {w: i for i, w in enumerate(words)}
    idx2word = {i: w for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for i, t in enumerate(tags)}

    return word2idx, idx2word, tag2idx, idx2tag


def convert_words_to_window(words, tags, window_size=5):
    """
    @param words: a list of lists of words in the sentences
    @param tags: a list of lists of tags in the sentences
    @param window_size: the size of the window to use
    @return: a list of tuples that return the full vector of window_size*emb_size and the actual tag
    """
    vector_out = []
    tag_out = []
    padding = (window_size - 1) // 2

    for sentence, sentence_tag in zip(words, tags):
        # Apply padding
        padded_sentence = ['<PAD_START>'] * padding + sentence + ['<PAD_END>'] * padding

        for i in range(padding, len(padded_sentence) - padding):
            # Extract the window
            window = padded_sentence[i - padding: i + padding + 1]
            vector_out.append(window)
            if tags is not None:
                tag_out.append(sentence_tag[i - padding])  # Adjust index for original sentence

    return vector_out, tag_out if tags is not None else vector_out


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
    for w, t in zip(window, tag):
        vector_out.append([word2idx[w] if w in word2idx else word2idx['<UNK>'] for w in w])
        tag_out.append(tag2idx[t])

    return vector_out, tag_out

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
    #test_words,_ = read_test_data('./../Data/pos/test')



    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(train_words, train_tags)
    windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=5)
    windows_idx, window_tags_idx = convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx)


    for i in range(10):
        print(windows[i], window_tags[i])

    # later
    # word2vec = word2vec(words, vectors)
    # vectors = read_vectors('../Data/wordVectors.txt')