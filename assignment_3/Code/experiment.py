"""
'experiment.py' file does as follows:
-------------------------------------
- Train and test a Recurrent Neural Network (RNN) acceptor that can preform
    binary classification on the input sequences.
- The RNN is specifically LSTM (Long Short-Term Memory) network followed by a
    MLP (Multi-Layer Perceptron) classifier with a single hidden layer.
- Formally, for an input sequence of n vectors x_1, x_2, ..., x_n,
    the output of the network is MLP(LSTM(x_1, x_2, ..., x_n)).
    where LSTM(x_1, x_2, ..., x_n) is mapping the sequence into a vector in R^d_1
    and MLP is used to classify this d_1-dimensional vector.


The language is defined as follows:
Positive examples:
------------------
[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+

Negative examples:
------------------
[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+
"""

# Importing necessary libraries
import os
import argparse
from random import random

import torch
import logging as log
from torch import nn

import utils


class LSTMAcceptor(nn.Module):
    """
    @brief: LSTM Acceptor class.
    LSTM (Long Short-Term Memory) network followed by a MLP
    (Multi-Layer Perceptron) classifier with a single hidden layer.
    """
    def __init__(self, vocab_size, lstm_input_size, lstm_hidden_size, mlp_hidden_size, mlp_output_size, device, padding_idx, dropout):
        """
        @brief: Initialize the LSTM Acceptor.
        @param vocab_size: The size of the vocabulary.
        @param lstm_input_size: The size of the input vectors.
        @param lstm_hidden_size: The size of the hidden state.
        @param mlp_hidden_size: The size of the hidden layer in the MLP.
        @param mlp_output_size: The size of the output layer in the MLP.
        @param device: Device for computation.
        @param padding_idx: The index of the padding token.
        @param dropout: Dropout probability.
        """
        super(LSTMAcceptor, self).__init__()
        self.vocab_size = vocab_size
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_output_size = mlp_output_size
        self.device = device
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=lstm_input_size, padding_idx=padding_idx)
        self.lstm = utils.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size, device=device, padding_idx=padding_idx)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size, out_features=mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_hidden_size, out_features=mlp_output_size),
            nn.Softmax(dim=1)
        )
        self.to(device)

    def forward(self, sequence):
        """
        @brief: Forward pass for the LSTM acceptor.
        @param sequence: Input sequence.
        @return: The output of the network.
        """
        # Get the original lengths of the input sequences
        original_lengths = (sequence != self.padding_idx).sum(dim=1)

        # Embed the input sequence
        embedded_sequence = self.embedding(sequence)

        # Forward pass through the LSTM
        lstm_output = self.lstm.acceptor_forward(sequence=embedded_sequence, original_lengths=original_lengths)

        # Forward pass through the MLP
        output = self.mlp(lstm_output)

        return output

    def predict(self, sequence):
        """
        @brief: Predict the class of a given sequence.
        @param sequence: Input sequence.
        @return: The predicted class.
        """
        with torch.no_grad():
            output = self.forward(sequence)
            _, predicted = torch.max(output, 1)
        return predicted


    def loss(self, y_pred, y_true):
        """
        @brief: Compute the loss of the network.
        @param y_pred: Predicted class.
        @param y_true: True class.
        @return: The loss of the network.
        """
        return nn.CrossEntropyLoss(reduction='sum')(y_pred, y_true)

    def get_model_metadata(self):
        """
        @brief: Get the metadata of the model.
        @return: Metadata of the model.
        """
        return {
            'vocab_size': self.vocab_size,
            'lstm_input_size': self.lstm_input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'mlp_hidden_size': self.mlp_hidden_size,
            'mlp_output_size': self.mlp_output_size,
            'device': self.device,
            'padding_idx': self.padding_idx
        }

def main(parsed_args):
    """
    @brief: Main function for the experiment.
    @param args: Arguments for the experiment.
    """
    if not parsed_args.seed:
        seed = random.randint(0, 2**31)
    utils.set_seed(parsed_args.seed)

    # Set the device for computation
    device = utils.get_device()

    # Train the model
    if parsed_args.train:
        # Load the data
        train_loader, dev_loader = utils.get_train_dev_data_loader(ds_type=utils.DatasetType.POS_NEG,
                                                                   data_filename=parsed_args.train_data_file,
                                                                   device=device, batch_size=parsed_args.batch_size,
                                                                   dev_ratio=parsed_args.dev_ratio)
        train_metadata = train_loader.dataset.get_metadata()

        # Initialize the model
        model = LSTMAcceptor(vocab_size=train_metadata['vocab_size'], lstm_input_size=parsed_args.embedding_dim,
                             lstm_hidden_size=parsed_args.lstm_hidden_dim, mlp_hidden_size=parsed_args.mlp_hidden_dim,
                             mlp_output_size=2, device=device, padding_idx=train_metadata['padding_token_idx'], dropout=parsed_args.dropout)
        log.info(f'Model: {model}')

        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=parsed_args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        trainer = utils.TorchTrainer(model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        fit_res = trainer.fit(dl_train=train_loader, dl_val=dev_loader, num_epochs=parsed_args.epochs, early_stopping=True)


        # Save the model
        utils.save_model(model, parsed_args.save_model_file)

    # Predict using the model
    if parsed_args.predict:
        # Load the data
        predict_loader = utils.get_predict_data_loader(parsed_args.data_file, parsed_args.batch_size, parsed_args.padding_idx, parsed_args.vocab_size, parsed_args.device)
        predict_metadata = predict_loader.dataset.get_metadata()

        # Load the model
        model = utils.load_model(parsed_args.load_model_file, device=device)
        log.info(f'Model: {model}')

        # Predict using the model
        predictions = utils.predict(model, predict_loader)
        utils.save_predictions(predictions, parsed_args.predict_output_file)

    return 0


def parse_cli():
    p = argparse.ArgumentParser(description='Train and test a Recurrent Neural Network (RNN) acceptor.')
    p.add_argument('--debug', '-d', type=bool, help='Enable debugging.',
                        default=False, required=False)
    p.add_argument('--train', type=bool, help='Train the model.',
                        default=True, required=False)
    p.add_argument('--predict', type=bool, help='Predict using the model.',
                        default=True, required=False)
    p.add_argument('--log', type=bool, help='Enable logging.',
                        default=True, required=False)
    p.add_argument('--log_dir', type=str, help='Directory to save logs.',
                        default='Logs', required=False)
    p.add_argument('--train_data_file', type=str, help='Training data file.',
                        default='Data/Pos_Neg_Examples/train', required=False)
    p.add_argument('--dev_data_file', type=str, help='Development data file.',
                        default='Data/Pos_Neg_Examples/test', required=False)
    p.add_argument('--predict_output_file', type=str, help='Predictions output file.',
                        default='outputs/predictions/pos_neg_examples.pred', required=False)
    p.add_argument('--save_model_file', type=str, help='Model output file.',
                        default='outputs/models/pos_neg_examples.pth', required=False)
    p.add_argument('--load_model_file', type=str, help='Model input file.',
                        default='outputs/models/pos_neg_examples.pth', required=False)
    p.add_argument('--dev_ratio', type=float, help='Development ratio.',
                        default=0.1, required=False)
    p.add_argument('--batch_size', type=int, help='Batch size.',
                        default=32, required=False)
    p.add_argument('--epochs', type=int, help='Number of epochs.',
                        default=15, required=False)
    p.add_argument('--embedding_dim', type=int, help='Embedding dimension.',
                        default=20, required=False)
    p.add_argument('--lstm_hidden_dim', type=int, help='LSTM hidden dimension.',
                        default=32, required=False)
    p.add_argument('--mlp_hidden_dim', type=int, help='MLP hidden dimension.',
                        default=16, required=False)
    p.add_argument('--lr', type=float, help='Learning rate.',
                        default=0.001, required=False)
    p.add_argument('--dropout', type=float, help='Dropout probability.',
                        default=0.5, required=False)
    p.add_argument('--seed', type=int, help='Random seed.',
                        default=42, required=False)

    return p.parse_args()


if __name__ == '__main__':
    if utils.is_debugging():
        parsed_args = argparse.Namespace(
            debug=True,
            train=True,
            predict=True,
            log=True,
            log_dir='Logs',
            train_data_file='Data/Pos_Neg_Examples/train',
            dev_data_file='Data/Pos_Neg_Examples/test',
            predict_output_file='outputs/predictions/pos_neg_examples.pred',
            save_model_file='outputs/models/pos_neg_examples.pth',
            load_model_file='outputs/models/pos_neg_examples.pth',
            dev_ratio=0.1,
            batch_size=32,
            epochs=15,
            embedding_dim=20,
            lstm_hidden_dim=32,
            mlp_hidden_dim=16,
            lr=0.001,
            dropout=0.5,
            seed=42,
        )
    else:
        parsed_args = parse_cli()
    log.basicConfig(level=log.INFO)
    main(parsed_args)








