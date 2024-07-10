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
import torch
from matplotlib import pyplot as plt

from utils import *
from tqdm import tqdm
import time
import numpy as np









def main():
    parser = argparse.ArgumentParser(description='BiLSTM tagger')
    parser.add_argument('--repr', type=str, help='The model to use must be a, b, c or d', default='a')
    parser.add_argument('--task', type=str, help='Task pos or ner', default='pos')
    parser.add_argument('--save_model', type=str, help='t for saving the model otherwise f', default='t')
    parser.add_argument('--hidden_size', type=int, help='The number of hidden units in the LSTM', default=32)
    parser.add_argument('--embedding_dim', type=int, help='The size of the word embeddings', default=32)
    parser.add_argument('--char_hidden_size', type=int, help='The size of the character LSTM hidden units', default=4)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=10)

    args = parser.parse_args()


    assert args.repr in ['a', 'b', 'c', 'd'], 'repr must be a, b, c or d, the given argument for repr is {}'.format(args.repr)

    task = args.task




    # load training data
    train_file = f'../Data/{task}/train'
    eval_file = f'../Data/{task}/dev'
    vocab_file = f'../Data/vocab.txt'
    files_dir= f'../Data/{task}'
    plot_file = f'../outputs/plots/{task}_{args.repr}.png'
    accuracies_file = f'../outputs/accuracies/{task}_{args.repr}.npy'
    word2idx, idx2word = read_all_words(vocab_file, files_dir)
    start_time = time.time()
    ds_train = TaggerDataset(train_file, vocab_file, files_dir, word2idx, idx2word, mode=args.repr)
    load_ds_train_time = time.time() - start_time
    ds_eval = TaggerDataset(eval_file, vocab_file, files_dir, word2idx, idx2word, max_len_train=ds_train.max_len, mode=args.repr)
    load_ds_eval_time = time.time() - load_ds_train_time

    print(f'Load dataset time: {load_ds_train_time} and load dataset time: {load_ds_eval_time}')


    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=64, shuffle=False)




    # initialize model
    model = get_model(args, ds_train)
    padding_idx = ds_train.word2idx[PAD_TAG]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



    # train model
    best_acc = 0

    accuracy_for_plot = []
    for epoch in range(args.epochs):

        epoch_loss = train_epoch(dl_train, model, optimizer, padding_idx)
        epoch_acc = test_epoch(dl_eval, epoch, model, padding_idx, task)
        acc = sum(epoch_acc) / len(epoch_acc)
        accuracy_for_plot.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'../outputs/models/part3/model_{args.repr}_{task}_best.pth')

    torch.save(model.state_dict(), f'../outputs/models/part3/model_{args.repr}_{task}_last.pth')

    plot_accuracies(args, accuracy_for_plot, plot_file)
    save_accuracies(accuracy_for_plot, accuracies_file)
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
            original_len=original_len
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


    return model


def test_epoch(dl_eval, epoch, model, padding_idx, task):
    accuracies = []

    with torch.no_grad():
        for (data, labels) in tqdm(dl_eval, desc=f'Evaluation {len(dl_eval)}'):
            output = model(data)
            n_correct = 0
            checked = 0
            output = output.argmax(dim=2) - 1
            for pred_line, true_line in zip(output, labels):
                for pred_tag, true_tag in zip(pred_line, true_line):
                    if true_tag != padding_idx:
                        if task == 'pos' or (task == "ner" and true_tag != dl_eval.dataset.tag2idx[PAD_TAG]):
                            n_correct += 1 if pred_tag == true_tag else 0
                            checked += 1


            accuracy = n_correct / checked

            accuracies.append(accuracy)
    acc = sum(accuracies) / len(accuracies)
    print(f"Epoch: {epoch}: Accuracy: {acc}")
    return accuracies


def train_epoch(dl_train, model, optimizer, padding_idx):
    losses = []
    accuracies = []
    t = tqdm(dl_train, desc=f'Training,{len(dl_train)}')
    i = 0
    for (data, labels) in t:
        optimizer.zero_grad()
        output = model(data)
        # print('Forward time:', forward_end - forward_start)
        loss = model.loss(output, labels, padding_idx)
        loss.backward()
        optimizer.step()
        losses.append(loss)

        t.set_description(f'Training,{len(dl_train)}, Loss = {sum(losses) / len(losses)}')
    return losses


def plot_accuracies(args, accuracies, file):
    plt.plot(accuracies)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.legend([f'Accuracy using representation ({args.repr})'])
    plt.title("Accuracy vs train steps")
    plt.savefig(file)
    plt.show()

def save_accuracies(accuracies, file):
    accuracies = np.array(accuracies)
    np.save(file, accuracies)




if __name__ == '__main__':
    main()