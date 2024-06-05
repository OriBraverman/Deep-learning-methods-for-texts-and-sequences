import numpy as np
import torch

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
    return {w: v for w, v in zip(words, vectors)}

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

if __name__ == '__main__':
    words, tags = read_data('Data/ner/train')
    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(words, tags)
    print(word2idx)
    print(idx2word)
    print(tag2idx)
    print(idx2tag)
