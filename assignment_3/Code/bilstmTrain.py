"""
Part 3: Implementing the BiLSTM Tagger
--------------------------------------

input sequence: w1, w2, w3, ..., wn
vector representation: xi = repr(wi)
biLSTM: bi = biLSTM(x1, ..., xi) = LSTM_F(x1, ..., xi) ◦ LSTM_B(xn, ..., xi)
biLSTM layer: b'i = biLSTM(b1, ..., bn;i)
label prediction: yi = softmax(linear(b'i)) (cross-entropy loss)

word representation options:
(a) embedding vector: repr(wi) = E[wi]
(b) character-level LSTM: repr(wi) = repr(c1, c2, ..., cmi) = LSTM_C(E[c1], ..., E[cmi])
(c) embeddings + subword representation from assignment 2
(d) concatenation of (a) and (b) followed by a linear layer
"""

import os
import sys

import argparse
import random
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt

from utils import *
from tqdm import tqdm
import time
import numpy as np


def main():
    parser: ArgumentParser = argparse.ArgumentParser(description='BiLSTM tagger')
    parser.add_argument('repr', type=str, help='The model to use must be a, b, c or d', default='b')
    parser.add_argument('trainFile', type=str, help='The training file')
    parser.add_argument('modelFile', type=str, help='The file for saving the model')
    parser.add_argument('--task', type=str, help='Task pos or ner', default='ner')
    parser.add_argument('--save_model', type=str, help='t for saving the model otherwise f', default='t')
    parser.add_argument('--hidden_size', type=int, help='The number of hidden units in the LSTM', default=32)
    parser.add_argument('--embedding_dim', type=int, help='The size of the word embeddings', default=32)
    parser.add_argument('--char_hidden_size', type=int, help='The size of the character LSTM hidden units', default=4)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)
    parser.add_argument('--lr', type=float, help='learning rate', default=0.001)

    args = parser.parse_args()

    assert args.repr in ['a', 'b', 'c', 'd'], 'repr must be a, b, c or d, the given argument for repr is {}'.format(
        args.repr)
    assert args.task in ['pos', 'ner'], 'task must be pos or ner, the given argument for task is {}'.format(args.task)
    task = args.task

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        mps_available = True
    else:
        mps_available = False
        device = 'cpu'

    # load training data
    train_file = f'../Data/{task}/train'
    eval_file = f'../Data/{task}/dev'
    vocab_file = f'../Data/vocab.txt'
    files_dir = f'../Data/{task}'
    plot_file = f'../outputs/plots/{task}_{args.repr}.png'
    accuracies_file = f'../outputs/accuracies/{task}_{args.repr}.npy'
    word2idx, idx2word = read_all_words(vocab_file, files_dir)

    ds_train = TaggerDataset(train_file, vocab_file, files_dir, word2idx, idx2word, mode=args.repr)
    ds_eval = TaggerDataset(eval_file, vocab_file, files_dir, word2idx, idx2word, max_len_train=ds_train.max_len,
                            mode=args.repr,
                            tag2idx=ds_train.tag2idx, idx2tag=ds_train.idx2tag)

    train_batch_size = 64
    eval_batch_size = 64

    dl_train = DataLoader(ds_train, batch_size=train_batch_size, shuffle=args.repr != 'd')
    dl_eval = DataLoader(ds_eval, batch_size=eval_batch_size, shuffle=False)

    print(ds_train.n_tags)

    if args.repr == 'd':
        ds_train_char = TaggerDataset(train_file, vocab_file, files_dir, word2idx, idx2word, mode='b')
        ds_eval_char = TaggerDataset(eval_file, vocab_file, files_dir, word2idx, idx2word,
                                     max_len_train=ds_train.max_len,

                                     mode='b', tag2idx=ds_train.tag2idx, idx2tag=ds_train.idx2tag)
        dl_train_char = DataLoader(ds_train_char, batch_size=train_batch_size, shuffle=args.repr != 'd')
        dl_eval_char = DataLoader(ds_eval_char, batch_size=eval_batch_size, shuffle=False)

        dl_train = [dl_train, dl_train_char]
        dl_eval = [dl_eval, dl_eval_char]

    # initialize model
    model = get_model(args, ds_train) if args.repr != 'd' else get_model(args, ds_train_char)
    padding_idx = ds_train.tag2idx[PAD_TAG]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    if mps_available:
        model = model.to(device)

    # train model
    best_acc = 0

    accuracy_for_plot = []
    for epoch in range(args.epochs):

        accuracies = train_epoch(dl_train, dl_eval, model, optimizer, padding_idx, args, is_char=args.repr in ['b', 'd'], device=device)
        acc = sum(accuracies[-70:]) / 70
        accuracy_for_plot.extend(accuracies)
        print(f'Epoch: {epoch+1}, Accuracy: {acc}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model.to(device='cpu').state_dict(),
                       f'../outputs/models/part3/model_{args.repr}_{task}_best.pth')
            model.to(device=device)

    torch.save(model.to(device='cpu').state_dict(), args.modelFile)


    plot_accuracies(args, accuracy_for_plot, plot_file)

def get_model(args, ds_train: TaggerDataset):
    if args.repr == 'a':
        input_size = ds_train.max_len  # input size
        hidden_size = args.hidden_size  # hidden size
        embedding_dim = args.embedding_dim
        output_size = ds_train.n_tags  # output size
        vocab_size = len(ds_train.word2idx)
        padding_idx = ds_train.word2idx[PAD_TAG]
        original_len = ds_train.max_len
        model = TaggerBiLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            padding_idx=padding_idx,
            original_len=original_len,
        )

    elif args.repr == 'b':

        hidden_size = args.hidden_size
        char_hidden_size = args.char_hidden_size
        embedding_dim = args.embedding_dim
        output_size = ds_train.n_tags
        ab_size = len(ds_train.word2idx)
        padding_idx = ds_train.word2idx[PAD_TAG]
        char_padding_idx = ds_train.letter2idx[PAD_LETTER_TAG]
        original_len = ds_train.max_len
        model = CharTaggerBiLSTM(
            ab_size=ab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            char_hidden_size=char_hidden_size,
            output_size=output_size,
            padding_idx=padding_idx,
            char_padding_idx=char_padding_idx,
            original_len=original_len
        )
    elif args.repr == 'c':
        input_size = ds_train.max_len  # input size
        hidden_size = args.hidden_size  # hidden size
        embedding_dim = args.embedding_dim
        output_size = ds_train.n_tags  # output size
        vocab_size = len(ds_train.word2idx)
        padding_idx = ds_train.word2idx[PAD_TAG]
        original_len = ds_train.max_len
        model = CBOWTagger(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            padding_idx=padding_idx,
            original_len=original_len,
            idx_word2idx_pre=ds_train.idx_word2idx_pre,
            idx_word2idx_suf=ds_train.idx_word2idx_suf
        )
    elif args.repr == "d":
        input_size = ds_train.max_len  # input size
        hidden_size = args.hidden_size  # hidden size
        embedding_dim = args.embedding_dim
        output_size = ds_train.n_tags  # output size
        vocab_size = len(ds_train.word2idx)
        padding_idx = ds_train.word2idx[PAD_TAG]
        original_len = ds_train.max_len
        char_padding_idx = ds_train.letter2idx[PAD_LETTER_TAG]
        ab_size = len(ds_train.idx2word)
        model = MaxLSTM(
            vocab_size=vocab_size,
            ab_size=ab_size,
            embedding_dim=embedding_dim,
            char_embedding_dim=4,
            hidden_size=hidden_size,
            output_size=output_size,
            padding_idx=padding_idx,
            char_padding_idx=char_padding_idx,
            original_len=original_len
        )

    return model


def test_epoch(dl_eval, model, padding_idx, task, repr, device='cpu'):
    accuracies = []

    with torch.no_grad():
        if repr != 'd':
            for (data, labels) in dl_eval:

                data, labels = data.to(device), labels.to(device)

                output = model(data)
                output = output.argmax(dim=2)

                pred, true = output.view(-1), labels.view(-1)

                mask = (true != padding_idx)

                if task == 'ner':
                    O_mask = true != dl_eval.dataset.tag2idx['O']
                    mask *= O_mask

                correct = pred.eq(true) * mask
                accuracy = correct.sum() / mask.sum()
                accuracies.append(accuracy.cpu().item())


        else:

            t = tqdm(zip(dl_eval[0], dl_eval[1]), desc='Training')
            for (data, labels), (data_char, _) in t:
                data, labels = data.to(device), labels.to(device)
                data_char = data_char.to(device)
                output = model(data_char, data)
                output = output.argmax(dim=2)

                pred, true = output.view(-1), labels.view(-1)

                mask = (true != padding_idx)

                if task == 'ner':
                    O_mask = true != dl_eval[0].dataset.tag2idx['O']
                    mask *= O_mask

                correct = pred.eq(true) * mask
                accuracy = correct.sum() / mask.sum()
                accuracies.append(accuracy.cpu().item())

    acc = sum(accuracies) / len(accuracies)
    #print(f"Epoch: {epoch}: Accuracy: {acc}")
    return acc


def train_epoch(dl_train, dl_eval, model, optimizer, padding_idx, args, is_char=False, device='cpu'):
    losses = []
    accuracies = []

    if args.repr != 'd':
        t = tqdm(dl_train, desc=f'Training')
        i = 0
        for (data, labels) in t:
            data = data.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            # print('Forward time:', forward_end - forward_start)
            loss = model.loss(output, labels, padding_idx)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            i += 1

            if i > 0 and i % 16 == 0:
                acc = test_epoch(dl_eval, model, padding_idx, args.task, args.repr, device)
                accuracies.append(acc)
                t.set_description(f'Training, Loss = {sum(losses) / len(losses)}, Accuracy = {acc}')



    else:
        t = tqdm(zip(dl_train[0], dl_train[1]), desc='Training')
        i = 0
        for (data, labels), (data_char, _) in t:
            data_char = data_char.to(device)
            labels = labels.to(device)
            data_char = data_char.to(device)
            optimizer.zero_grad()
            output = model(data_char, data)
            loss = model.loss(output, labels, padding_idx)
            loss.backward()
            optimizer.step()
            losses.append(loss)


            i += 1
            if i > 0 and i % 30 == 0:
                acc = test_epoch(dl_eval, model, padding_idx, args.task, args.repr, device)
                accuracies.append(acc)
                t.set_description(f'Training, Loss = {sum(losses) / len(losses)}, Accuracy = {acc}')

    return accuracies


def plot_accuracies(args, accuracies, file):
    print(accuracies)
    plt.plot(accuracies)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend([f'Accuracy using representation ({args.repr})'])
    plt.title("Accuracy vs train steps")
    plt.savefig(file)
    plt.show()




if __name__ == '__main__':
    main()
