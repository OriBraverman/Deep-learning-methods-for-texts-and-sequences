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

import argparse
import random
import torch

from assignment_3.Code import utils


class BiLSTMTagger(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTMTagger, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_f = torch.nn.LSTM(input_size, hidden_size)
        self.lstm_b = torch.nn.LSTM(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size * 2, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        h_f, _ = self.lstm_f(x)
        h_b, _ = self.lstm_b(x)
        h = torch.cat((h_f, h_b), 1)
        h, _ = self.lstm(h)
        h = self.linear(h)
        return self.softmax(h)  # cross-entropy loss


def main():
    parser = argparse.ArgumentParser(description='BiLSTM tagger')
    parser.add_argument('repr', type=str, help='input representation (a, b, c, d)')
    parser.add_argument('trainFile', type=str, help='input file to train on')
    parser.add_argument('modelFile', type=str, help='file to save/load the model')
    args = parser.parse_args()

    repr = args.repr
    trainFile = args.trainFile
    modelFile = args.modelFile

    # load training data
    with open(trainFile, 'r') as f:
        train_data = f.readlines()

    # initialize model
    input_size = 100  # input size
    hidden_size = 100  # hidden size
    output_size = 10  # output size
    model = BiLSTMTagger(input_size, hidden_size, output_size)

    # train model
    for epoch in range(5):
        for i, sentence in enumerate(train_data):
            # forward pass


if __name__ == '__main__':
    if utils.is_debugging():
        task = 'pos'
        parsed_args = argparse.Namespace(
            train=True,
            predict=True,
            log=True,
            log_dir='Logs',
            train_data_file=f'Data/{task}/train',
            dev_data_file=f'Data/{task}/dev',
            test_data_file=f'Data/{task}/test',
            predict_output_file=f'outputs/predictions/{task}.pred',
            save_model_file=f'outputs/models/{task}.pth',
            load_model_file=f'outputs/models/{task}.pth',
            plot_file=f'outputs/plots/{task}.png',
            dev_ratio=0.1,
            batch_size=16,
            epochs=25,
            embedding_dim=30,
            lstm_hidden_dim=32,
            mlp_hidden_dim=16,
            lr=0.003,
            dropout=0,
            weight_decay=0,
            seed=23,
        )
    else:
        parsed_args = parse_cli()
    log.basicConfig(level=log.INFO)
    main(parsed_args)