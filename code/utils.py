# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
import random
from collections import Counter

def read_data(fname):
    data = []
    for line in open(f'../data/{fname}', encoding='utf-8'):
        label, text = line.strip().lower().split("\t",1)
        data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return list(text)

fc_uni = Counter()
fc_bi = Counter()

TRAIN_UNI = [(l,text_to_unigrams(t)) for l,t in read_data("train")]
TRAIN_BI = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
DEV_UNI   = [(l,text_to_unigrams(t)) for l,t in read_data("dev")]
DEV_BI   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]


for l,feats in TRAIN_UNI:
    fc_uni.update(feats)

for l,feats in TRAIN_BI:
    fc_bi.update(feats)

# 600 most common bigrams in the training set.
vocab_uni = set([x for x,c in fc_uni.most_common(600)])
vocab_bi = set([x for x,c in fc_bi.most_common(600)])

# label strings to IDs
L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN_BI]))) )}
# feature strings (bigrams) to IDs
F2I_UNI = {f:i for i,f in enumerate(list(sorted(vocab_uni)) )}
F2I_BI = {f:i for i,f in enumerate(list(sorted(vocab_bi)) )}
