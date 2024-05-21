# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
def read_data(fname):
    data = []
    for line in open(f'../data/{fname}', encoding='utf-8'):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return ['%s' % c for c in text]

TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]
TEST = [(l,text_to_bigrams(t)) for l,t in read_data("test")]

from collections import Counter
fc = Counter()
for l,feats in TRAIN:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
# label IDs to strings
I2L = {i: l for l, i in L2I.items()}
# feature strings (bigrams) to IDs
F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

TRAIN_UNI = [(l,text_to_unigrams(t)) for l,t in read_data("train")]
DEV_UNI   = [(l,text_to_unigrams(t)) for l,t in read_data("dev")]


fc = Counter()
for l,feats in TRAIN_UNI:
    fc.update(feats)

# 600 most common bigrams in the training set.
vocab_UNI = set([x for x,c in fc.most_common(600)])

# label strings to IDs
L2I_UNI = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN_UNI]))))}
# feature strings (bigrams) to IDs
F2I_UNI = {f:i for i,f in enumerate(list(sorted(vocab_UNI)))}


#print(len(vocab_UNI))
