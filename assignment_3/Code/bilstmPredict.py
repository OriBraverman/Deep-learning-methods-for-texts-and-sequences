import argparse

import tqdm
import json

from utils import *
import torch
import torch.nn as nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repr', type=str, default='a')
    parser.add_argument('--modelFile', type=str, default='../outputs/models/part3/model_a_pos_best.pth')
    parser.add_argument('--inputFile', type=str, default='../Data/pos/test')


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    task = 'pos'
    args = parser.parse_args()
    train_file = f'../Data/{task}/train'
    eval_file = f'../Data/{task}/dev'
    test_file = f'../Data/{task}/test'
    vocab_file = f'../Data/vocab.txt'
    files_dir = f'../Data/{task}'
    word2idx, idx2word = read_all_words(vocab_file, files_dir)

    assert args.repr in ['a', 'b', 'c', 'd']

    ds_train = TaggerDataset(train_file, vocab_file, files_dir, word2idx, idx2word, mode=args.repr)
    ds_test = TaggerDataset(test_file, vocab_file, files_dir, word2idx, idx2word, max_len_train=ds_train.max_len,
                            mode=args.repr, test=True,
                            tag2idx=ds_train.tag2idx, idx2tag=ds_train.idx2tag)

    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False)

    json.dump(ds_train.tag2idx, open('../Data/tag2idx.json', 'w'))
    json.dump(ds_train.idx2tag, open('../Data/idx2tag.json', 'w'))




    vocab_size = len(ds_train.word2idx)
    output_size = ds_train.n_tags
    padding_idx = ds_train.word2idx[PAD_TAG]
    max_len = ds_train.max_len


    if args.repr == 'a':
        model = TaggerBiLSTM(vocab_size, 32, 32, output_size, padding_idx, max_len)
        model.load_state_dict(torch.load(args.modelFile))

    if args.repr == 'c':
        model = CBOWTagger(vocab_size, 32, 32, output_size, padding_idx, max_len, ds_train.idx_word2idx_suf, ds_train.idx_word2idx_pre)
        model.load_state_dict(torch.load(args.modelFile))

    words = []
    res = []
    true = []

    model.to(device)
    i = 0
    with torch.no_grad():
        for data, _ in tqdm.tqdm(dl_test):
            words.append([ds_train.idx2word[i.item()] for i in data.view(-1) if i.item() != ds_train.word2idx[PAD_TAG]])
            data = data.to(device)
            output = model(data)
            output = output.argmax(dim=2).cpu()
            res.append([t.item() for t in output.view(-1)][:len(words[-1])])

            if i == 100:
                break
            else:
                i+= 1


    couille = 0
    for i in range(len(words)):
        for j in range(len(words[i])):
            print(words[i][j], ds_train.idx2tag[res[i][j]])











