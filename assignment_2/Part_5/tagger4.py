import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from assignment_2.utils import *


TASK = 'pos'
TRAIN = True


class CharCNN(nn.Module):
    def __init__(self, char_embedding, num_filters, window_size, max_word_length):
        super(CharCNN, self).__init__()

        self.char_embedding - char_embedding
        self.char_embedding_dim = char_embedding.shape[1]
        self.max_word_length = max_word_length

        input_dim = self.char_embedding_dim
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=num_filters, kernel_size=window_size, stride=self.max_word_length)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.char_embedding(x)
        x = x.view(x.size(0), 1, self.max_word_length, self.char_embedding_dim)
        x = self.conv1(x)
        x = self.activation(x)
        x = F.max_pool2d(x, kernel_size=(x.size(2), 1))
        x = x.view(x.size(0), -1)
        return x

class Tagger4(nn.Module):
    def __init__(self, vocab_size, embedding, char_embedding, hidden_dim, output_dim, embedding_dim=50, window_size=5):
        super(Tagger4, self).__init__()

        # Embedding layer - 50 dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        # Fully connected
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        self.char_cnn = CharCNN(char_embedding, num_filters=50, window_size=5, max_word_length=20)

    def forward(self, x):
        x = self.embedding(x)
        # Flatten the tensor to 1D
        x = x.view(x.size(0), -1)
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
            for words, tags in tqdm(train_data, desc=f'Epoch {epoch + 1}/{epochs}'):
                # Zero the gradients before the backward pass
                optimizer.zero_grad()
                # Move the data to the device
                words, tags = torch.tensor(words).to(device), torch.tensor(tags).to(device)
                # Forward pass
                # first use the char_cnn to get the char embeddings
                char_embeddings = self.char_cnn(words)
                # then concatenate the char embeddings with the word embeddings
                words = torch.cat([words, char_embeddings], dim=1)
                output = self(words)
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

        return dev_loss_list, dev_accuracy_list

    def evaluate(self, data, idx2tag, device='cpu', is_ner=False):
        correct, total = 0, 0
        total_loss = 0
        with torch.no_grad():
            for words, tags in data:
                words, tags = torch.tensor(words).to(device), torch.tensor(tags).to(device)
                output = self(words)
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

    def predict(self, data, idx2tag, save_file, device='cpu'):
        predictions = []
        for words, _ in tqdm(data):
            # Move the data to the device
            words = torch.tensor(words).to(device)
            # Forward pass
            output = self(words)

            pred = output.argmax(dim=1)

            pred_tag = [idx2tag[p] for p in pred.tolist()]
            predictions.extend(pred_tag)


        with open(save_file, 'w') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')

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
        return pattern if pattern in word2idx else num_pattern
    elif all([c.isdigit() or c in [','] for c in word]):
        return num_pattern
    else:
        return None


def convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx):
    windows_idx = []
    window_tags_idx = []
    for window, window_tag in zip(windows, window_tags):
        window_idx = [convert_word_to_idx(word, word2idx) for word in window]
        window_tag_idx = [tag2idx[tag] for tag in window_tag]
        windows_idx.append(window_idx)
        window_tags_idx.append(window_tag_idx)
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


if __name__ == '__main__':
    vecs = np.loadtxt("../Data/wordVectors.txt")

    with open("../Data/vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]

    word2idx, idx2word = make_word_dict(vocab)

    """     TRAIN    """
    train_words, train_tags = read_data(f'Data/{TASK}/train')
    # Create the vocabularies
    _, _, tag2idx, idx2tag = make_vocabs(train_words, train_tags)

    vocab_size = len(word2idx)
    output_dim = len(tag2idx)
    hidden_dim = 32
    n_epoch = 25

    if TRAIN:
        # Create an output directory in which to save the generated files
        if not os.path.exists('Output'):
            os.makedirs('Output')

        # Convert the words to windows
        windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=5)
        # Convert the windows to indices
        windows_idx, window_tags_idx = convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx)

        # Cut the data to 1000 samples in order to debug faster
        windows_idx = windows_idx
        window_tags_idx = window_tags_idx



        batch_size = 64
        model = Tagger4(vocab_size=vocab_size, embedding=vecs, hidden_dim=hidden_dim, output_dim=output_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # TODO SAVE THE DATASET TO SAVE TIME AT EACH RUN
        train_data = TensorDataset(torch.tensor(windows_idx), torch.tensor(window_tags_idx))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        print(len(train_data))
        dev_words, dev_tags = read_data(f'Data/{TASK}/dev')
        dev_windows, dev_window_tags = convert_words_to_window(dev_words, dev_tags, window_size=5)
        dev_windows_idx, dev_window_tags_idx = convert_window_to_window_idx(dev_windows, dev_window_tags, word2idx, tag2idx)
        dev_data = TensorDataset(torch.tensor(dev_windows_idx), torch.tensor(dev_window_tags_idx))
        dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)

        dev_loss_list, dev_accuracy_list = model.train(optimizer, train_dataloader, dev_dataloader, idx2tag, epochs=n_epoch,
                                                       is_ner=TASK == 'ner')

        make_graph(dev_loss_list, 'Loss over epochs', 'Loss', 'Output/loss.png')
        make_graph(dev_accuracy_list, 'Accuracy over epochs', 'Accuracy', 'Output/accuracy.png')

        torch.save(model.state_dict(), f'model_part4_{TASK}.pth')

    tag2idx['<TEST>'] = len(tag2idx)

    test_words, test_tags = read_test_data(f'Data/{TASK}/test')
    test_windows, test_window_tags = convert_words_to_window(test_words, test_tags, window_size=5)
    test_windows_idx, test_window_tags_idx = convert_window_to_window_idx(test_windows, test_window_tags, word2idx, tag2idx)
    test_data = TensorDataset(torch.tensor(test_windows_idx), torch.tensor(test_window_tags_idx))
    test_dataloader = DataLoader(test_data, batch_size=128)

    model = Tagger4(vocab_size=vocab_size, embedding=vecs, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(f'model_part4_{TASK}.pth'))
    model.predict(test_dataloader, idx2tag, f'Output/part4.{TASK}')

