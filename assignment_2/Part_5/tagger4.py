import sys
from datetime import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from assignment_2.utils import *

# Constants
PART = 'Part_5'
TASK = 'ner'
TRAIN = True
TRAIN_BATCH_SIZE = 16 if TASK == 'pos' else 32
DEV_BATCH_SIZE = 16 if TASK == 'pos' else 32


class CharCNN(nn.Module):
    def __init__(self, char_embedding, num_filters, window_size, max_word_length):
        super(CharCNN, self).__init__()

        self.char_embedding = char_embedding
        self.char_embedding_dim = char_embedding.embedding_dim
        self.max_word_length = max_word_length

        input_dim = self.char_embedding_dim
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=(window_size, input_dim))
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.char_embedding(x)
        batch_size, max_word_length, embedding_dim = x.size()

        # Reshape for convolutional layer: [batch_size, 1, max_word_length, embedding_dim]
        x = x.view(batch_size, 1, max_word_length, embedding_dim)
        x = self.conv1(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(x.size(2), 1))
        x = x.view(x.size(0), -1)
        return x

class Tagger4(nn.Module):
    def __init__(self, vocab_size, embedding, char_embedding, hidden_dim, output_dim, max_word_len, num_filters, embedding_dim=50, window_size=5, char_window=5):
        super(Tagger4, self).__init__()
        input_dim = num_filters + embedding_dim * window_size

        # Embedding layer - 50 dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        # Fully connected
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        self.char_embedding = char_embedding

        self.char_cnn = CharCNN(char_embedding, num_filters=num_filters, window_size=char_window, max_word_length=max_word_len)

    def forward(self, words, chars):
        x = self.embedding(words)
        char_features = self.char_cnn(chars)
        # Flatten the tensor to 1D
        x = x.view(x.size(0), -1)
        x = torch.cat((x, char_features), dim=1)
        x = F.tanh(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def train(self, optimizer, train_data, dev_data, idx2tag, device='cpu', epochs=10, is_ner=False):
        dev_loss_list, dev_accuracy_list = [], []
        # Move the model to the device
        self.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for words, tags, chars in tqdm(train_data, desc=f'Epoch {epoch + 1}/{epochs}'):
                # Zero the gradients before the backward pass
                optimizer.zero_grad()
                # Move the data to the device
                words, tags, chars = torch.tensor(words).to(device), torch.tensor(tags).to(device), torch.tensor(chars).to(device)
                # Forward pass
                output = self(words, chars)
                # Compute the loss
                loss = self.loss_function(output, tags)
                total_loss += loss.item()
                # Backward pass
                loss.backward()
                # Update the weights
                optimizer.step()
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_data)
            # Evaluate the model on the training data
            train_accuracy, _ = self.evaluate(train_data, idx2tag, device, is_ner=is_ner)
            # Evaluate the model on the dev data
            dev_accuracy, dev_loss = self.evaluate(dev_data, idx2tag, device, is_ner=is_ner)
            # Save the dev loss and accuracy
            dev_loss_list.append(dev_loss)
            dev_accuracy_list.append(dev_accuracy)

            print(
                f'Epoch {epoch + 1}/{epochs} - Avg. Loss: {avg_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Dev Accuracy: {dev_accuracy:.4f} - Dev Loss: {dev_loss:.4f}')

            # Early stopping
            if epoch > 0 and dev_loss_list[-1] > dev_loss_list[-2]:
                print('Early stopping')
                break

        return dev_loss_list, dev_accuracy_list

    def evaluate(self, data, idx2tag, device='cpu', is_ner=False):
        correct, total = 0, 0
        total_loss = 0
        with torch.no_grad():
            for words, tags, chars in data:
                words, tags, chars = torch.tensor(words).to(device), torch.tensor(tags).to(device), torch.tensor(chars).to(device)
                output = self(words, chars)
                loss = self.loss_function(output, tags)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                if is_ner:
                    # ignore all cases where both the predicted and the actual tag are 'O'
                    for p, t in zip(predicted, tags):
                        if idx2tag[p.item()] != 'O' or idx2tag[t.item()] != 'O':
                            correct += (p == t).sum().item()
                            total += 1
                else:
                    correct += (predicted == tags).sum().item()
                    total += len(tags)
        return correct / total, total_loss / len(data.dataset)

    def parameters(self, recurse: bool = True, use_embeddings: bool = True):
        if use_embeddings:
            return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.embedding.parameters())
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def predict(self, windows_idx, original_data, test_chars, idx2tag, save_file, device='cpu'):

        # if file already exists, delete it
        if os.path.exists(save_file):
            os.remove(save_file)
        f = open(save_file, 'w')
        Predicted_tags = []
        with torch.no_grad():
            for words, chars in zip(windows_idx, test_chars):
                words = torch.tensor(words).to(device).unsqueeze(0)
                chars = torch.tensor(chars).to(device).unsqueeze(0)
                output = self(words, chars)
                _, predicted = torch.max(output.data, 1)
                Predicted_tags.append(predicted.item())
        i = 0
        for sentence in original_data:
            for word in sentence:
                f.write(f'{word} {idx2tag[Predicted_tags[i]]}\n')
                i += 1
            f.write('\n')
        f.close()

def make_word_dict(vocab):
    word2idx = {word: i for i, word in enumerate(list(sorted(set(vocab))))}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word

def make_tag_dict(tags):
    tag2idx = {tag: i for i, tag in enumerate(list(sorted(set([tag for tags in tags for tag in tags]))))}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    return tag2idx, idx2tag


def convert_to_number(word, word2idx):
    """
    description: convert to a pattern of numbers in vocab.txt
    If the pattern not in the vocab.txt, return 'NNNUMMM'
    The patterns are:
    - DG: for any digit
    - DG.DG: for any digit with a dot
    - .DG: for any digit starting with a dot
    - -DG: for any digit starting with a minus
    - +DG: for any digit starting with a plus
    """
    num_pattern = 'NNNUMMM'
    # First check if the word is composed of [digits/./-/+] only
    if all([c.isdigit() or c in ['.', '-', '+'] for c in word]):
        pattern = ['DG' if ch.isdigit() else ch for ch in word]
        pattern = ''.join(pattern)
        return pattern if pattern in word2idx else num_pattern
    elif all([c.isdigit() or c in [','] for c in word]):
        return num_pattern
    else:
        return None


def convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx):
    windows_idx = []
    window_tags_idx = []
    if window_tags is None:
        for window in windows:
            window_idx = [convert_word_to_idx(word, word2idx) for word in window]
            windows_idx.append(window_idx)
        return windows_idx
    for window, tag in zip(windows, window_tags):
        window_idx = [convert_word_to_idx(word, word2idx) for word in window]
        windows_idx.append(window_idx)
        window_tags_idx.append(tag2idx[tag])
    return windows_idx, window_tags_idx


def convert_word_to_idx(word, word2idx):
    unknown_vector = 'UUUNKKK'
    if word in word2idx:
        return word2idx[word]
    elif word.lower() in word2idx:
        return word2idx[word.lower()]
    elif convert_to_number(word, word2idx) in word2idx:
        return word2idx[convert_to_number(word, word2idx)]
    else:
        return word2idx[unknown_vector]


def convert_words_to_ord_char(words, max_word_len, window_size, pad_char=' '):
    chars = []
    # the output size is pad_size * 2 + max_word_len
    # for example, if the max_word_len is 10 and the window_size is 5, the pad_size is 2 => 2 * 2 + 10 = 14
    # for the word 'hello', the output will be [32, 32, 104, 101, 108, 108, 111, 32, 32, 32, 32, 32, 32, 32]
    pad_size = window_size // 2
    for sentence in words:
        for word in sentence:
            # max_word_len + 2 * pad_size - len(word) - pad_size = max_word_len - len(word) + pad_size
            word = pad_char * pad_size + word + pad_char * (max_word_len - len(word) + pad_size)
            chars.append([ord(char) for char in word])
    return chars



if __name__ == '__main__':
    # Save the output of the screen to a file
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sys.stdout = open(f'{PART}/Output/output_{TASK}_{time}.txt', 'w')
    # Define hyperparameters for experiments
    hyperparams = [
        {'num_filters': 30, 'window_size': 3, 'train': True},  # Baseline as described in the paper: Ma and Hovy 2016
        {'num_filters': 10, 'window_size': 3, 'train': True},
        {'num_filters': 20, 'window_size': 3, 'train': True},
        {'num_filters': 50, 'window_size': 3, 'train': True},
        {'num_filters': 100, 'window_size': 3, 'train': True},
        {'num_filters': 30, 'window_size': 2, 'train': True},
        {'num_filters': 30, 'window_size': 4, 'train': True},
        {'num_filters': 30, 'window_size': 5, 'train': True},
    ]


    vecs = np.loadtxt("Data/wordVectors.txt")

    with open("Data/vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]

    word2idx, idx2word = make_word_dict(vocab)

    """     TRAIN    """
    train_words, train_tags, max_word_len = read_data(f'Data/{TASK}/train', return_max_length=True)
    # Create the vocabularies
    _, _, tag2idx, idx2tag = make_vocabs(train_words, train_tags)

    vocab_size = len(word2idx)
    output_dim = len(tag2idx)
    hidden_dim = 32
    n_epoch = 25

    # Initialize the char embeddings matrix and dataset
    char_embedding = nn.Embedding(256, 30, padding_idx=0)
    nn.init.uniform_(char_embedding.weight, -np.sqrt(3 / 30), np.sqrt(3 / 30))

    if TRAIN:
        # Create an output directory in which to save the generated files
        if not os.path.exists(f'{PART}/Output'):
            os.makedirs(f'{PART}/Output')
        for hyperparam in hyperparams:

            if not hyperparam['train']:
                continue
            print(f'Training with hyperparameters: {hyperparam}')
            train_chars = convert_words_to_ord_char(train_words, max_word_len, window_size=hyperparam['window_size'])
            # Convert the words to windows
            windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=5)
            # Convert the windows to indices
            windows_idx, window_tags_idx = convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx)

            # Cut the data to 1000 samples in order to debug faster
            #windows_idx = windows_idx[:1000]
            #window_tags_idx = window_tags_idx[:1000]
            #train_chars = train_chars[:1000]

            model = Tagger4(vocab_size, vecs, char_embedding, hidden_dim, output_dim, max_word_len + 2*hyperparam['window_size'], num_filters=hyperparam['num_filters'], char_window=hyperparam['window_size'])
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            train_data = TensorDataset(torch.tensor(windows_idx), torch.tensor(window_tags_idx), torch.tensor(train_chars))
            train_dataloader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            # save the dataset
            # torch.save(train_data, f'dataset_tagger1_{TASK}.pth')

            print(len(train_data))
            dev_words, dev_tags = read_data(f'Data/{TASK}/dev')
            dev_chars = convert_words_to_ord_char(dev_words, max_word_len, window_size=hyperparam['window_size'])
            dev_windows, dev_window_tags = convert_words_to_window(dev_words, dev_tags, window_size=5)
            dev_windows_idx, dev_window_tags_idx = convert_window_to_window_idx(dev_windows, dev_window_tags, word2idx,
                                                                                tag2idx)
            dev_data = TensorDataset(torch.tensor(dev_windows_idx), torch.tensor(dev_window_tags_idx), torch.tensor(dev_chars))
            dev_dataloader = DataLoader(dev_data, batch_size=DEV_BATCH_SIZE, shuffle=True)

            dev_loss_list, dev_accuracy_list = model.train(optimizer, train_dataloader, dev_dataloader, idx2tag,
                                                           epochs=n_epoch,
                                                           is_ner=TASK == 'ner')

            make_graph(dev_loss_list, 'Loss over epochs', 'Loss', f'{PART}/Output/loss_{TASK}_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.png')
            make_graph(dev_accuracy_list, 'Accuracy over epochs', 'Accuracy', f'{PART}/Output/accuracy_{TASK}_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.png')

            torch.save(model.state_dict(), f'model_{PART}_{TASK}_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.pth')

    tag2idx['<TEST>'] = len(tag2idx)

    for hyperparam in hyperparams:
        if not os.path.exists(f'model_{PART}_{TASK}_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.pth'):
            continue
        model = Tagger4(vocab_size, vecs, char_embedding, hidden_dim, output_dim, max_word_len + 2*hyperparam['window_size'], num_filters=hyperparam['num_filters'], char_window=hyperparam['window_size'])
        model.load_state_dict(torch.load(f'model_{PART}_{TASK}_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.pth'))

        test_words = read_test_data(f'Data/{TASK}/test')
        test_windows = convert_words_to_window(test_words, window_size=5)
        test_windows_idx = convert_window_to_window_idx(test_windows, None, word2idx, tag2idx)
        test_chars = convert_words_to_ord_char(test_words, max_word_len, window_size=hyperparam['window_size'])

        model.predict(test_windows_idx, test_words, test_chars, idx2tag, f'{PART}/Output/test5_{hyperparam["num_filters"]}_{hyperparam["window_size"]}.{TASK}')

    sys.stdout.close()
