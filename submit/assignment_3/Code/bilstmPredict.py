import argparse

import tqdm
import json

from utils import *
import torch
import torch.nn as nn


if __name__ == '__main__':




    parser = argparse.ArgumentParser()
    parser.add_argument('repr', type=str, default='a')
    parser.add_argument('modelFile', type=str)
    parser.add_argument('inputFile', type=str)
    parser.add_argument('--task', type=str, default='pos')


    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    task = 'ner'
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


    vocab_size = len(ds_train.word2idx)
    output_size = ds_train.n_tags
    padding_idx = ds_train.word2idx[PAD_TAG]
    max_len = ds_train.max_len


    if args.repr == 'a':
        model = TaggerBiLSTM(vocab_size, 32, 32, output_size, padding_idx, max_len)
        model.load_state_dict(torch.load(args.modelFile))

    if args.repr == 'b':
        model = CharTaggerBiLSTM(len(ds_train.letter2idx), 32, 32, 4, output_size,
                                 padding_idx, ds_train.letter2idx[PAD_LETTER_TAG], max_len)

    if args.repr == 'c':
        model = CBOWTagger(vocab_size, 32, 32, output_size, padding_idx, max_len, ds_train.idx_word2idx_suf, ds_train.idx_word2idx_pre)
        model.load_state_dict(torch.load(args.modelFile))

    if args.repr == 'd':
        ds_test_char = TaggerDataset(train_file, vocab_file,files_dir, word2idx, idx2word, True, ds_train.max_len,
                                     'b', ds_train.tag2idx, ds_train.idx2tag)
        dl_test_char = DataLoader(ds_test_char, batch_size=1, shuffle=False)

        model = MaxLSTM(vocab_size, ds_test_char.max_word_len, 32, 32, output_size, padding_idx,
                        padding_idx, ds_train.letter2idx[PAD_LETTER_TAG], max_len)



    words = []
    res = []
    true = []

    model.to(device)
    i = 0
    with torch.no_grad():
        for data, data2 in tqdm.tqdm(dl_test):
            words.append([ds_train.idx2word[i.item()] for i in data.view(-1) if i.item() != ds_train.word2idx[PAD_TAG]])
            if args.repr != 'd':
                data = data.to(device)
                output = model(data)
            else:
                data_word = data[0].to(device)
                data_char = data2[0].to(device)
                output = model(data_word, data_char)
            output = output.argmax(dim=2).cpu()
            res.append([t.item() for t in output.view(-1)][:len(words[-1])])

    with open(f'../outputs/predictions/test4.{task}', 'w') as f:
        for line in res:
            for tag in line:
                f.write(ds_train.idx2tag[tag]+'\n')
            f.write('\n')












