import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.lm import Vocabulary
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
# import the utils.py file from the father directory
from assignment_2.utils import *

# Constants
TASK = 'pos'
TRAIN = True


class Tagger3(nn.Module):
    def __init__(self, vocab_size, pre_suf_size, hidden_dim, output_dim, embedding_dim=50, window_size=1):
        super(Tagger3, self).__init__()

        # Embedding layer - 50 dimensions
        word2vec = load_word2vec()
        word2vec = np.stack(list(word2vec.values()))

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))

        self.pre_suff_embedding = nn.Embedding(pre_suf_size, embedding_dim)
        # Fully connected
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x[:, 0]) + self.pre_suff_embedding(x[:, 1]) + self.pre_suff_embedding(x[:, 2])
        # Flatten the tensor to 1D
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def train(self, optimizer, train_data, dev_data, idx2tag, device='cpu', epochs=10, is_ner=False):
        dev_loss_list, dev_accuracy_list = [], []
        # Move the model to the device
        self.to(device)

        for epoch in range(epochs):
            total_loss = 0
            for word, tags in tqdm(train_data, desc=f'Epoch {epoch + 1}/{epochs}'):
                # Zero the gradients before the backward pass
                optimizer.zero_grad()
                # Move the data to the device
                words, tags = torch.tensor(word).to(device), torch.tensor(tags).to(device)

                # Forward pass
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

    def predict(self, data, idx2tag, device='cpu'):
        pass

    def parameters(self, recurse: bool = True, use_embeddings: bool = True):
        if use_embeddings:
            return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(
                self.pre_suff_embedding.parameters())
        return list(self.fc1.parameters()) + list(self.fc2.parameters())


if __name__ == '__main__':
    vocab = read_vocab()
    train_words, train_prefixes, train_suffixes, train_tags = read_data_pre_suf(f'Data/{TASK}/train')
    # Create the vocabularies
    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(vocab, train_tags)

    pre_suf_list = get_pre_suf_list(train_prefixes, train_suffixes)
    pre_suf2idx = make_pre_suf(pre_suf_list)

    vocab_size = len(word2idx)
    pre_suf_size = len(pre_suf2idx)
    output_dim = len(tag2idx)
    hidden_dim = 32
    n_epoch = 25
    batch_size = 64

    if TRAIN:
        # Create an output directory in which to save the generated files
        if not os.path.exists('Output'):
            os.makedirs('Output')

        # Convert the words to windows
        windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=1)
        # Convert the windows to indices
        windows_idx, window_tags_idx = convert_window_to_window_idx_presuf(windows, window_tags, word2idx, pre_suf2idx,
                                                                           tag2idx)

        model = Tagger3(vocab_size, pre_suf_size, hidden_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # TODO SAVE THE DATASET TO SAVE TIME AT EACH RUN
        train_data = TensorDataset(torch.tensor(windows_idx), torch.tensor(window_tags_idx))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        dev_words, dev_prefixes, dev_suffixes, dev_tags = read_data_pre_suf(f'Data/{TASK}/dev')

        dev_windows, dev_window_tags = convert_words_to_window(dev_words, dev_tags, window_size=1)
        # Convert the windows to indices
        dev_windows_idx, dev_window_tags_idx = convert_window_to_window_idx_presuf(windows, window_tags, word2idx,
                                                                                   pre_suf2idx,
                                                                                   tag2idx)
        dev_data = TensorDataset(torch.tensor(dev_windows_idx), torch.tensor(dev_window_tags_idx))
        dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)

        dev_loss_list, dev_accuracy_list = model.train(optimizer, train_dataloader, dev_dataloader, idx2tag,
                                                       epochs=n_epoch,
                                                       is_ner=TASK == 'ner')

        make_graph(dev_loss_list, 'Loss over epochs', 'Loss', 'Output/loss.png')
        make_graph(dev_accuracy_list, 'Accuracy over epochs', 'Accuracy', 'Output/accuracy.png')

        torch.save(model.state_dict(), f'model_part1_{TASK}.pth')

    tag2idx['<TEST>'] = len(tag2idx)

    test_words, test_tags = read_test_data(f'Data/{TASK}/test')
    test_windows, test_window_tags = convert_words_to_window(test_words, test_tags, window_size=5)
    test_windows_idx, test_window_tags_idx = convert_window_to_window_idx(test_windows, test_window_tags, word2idx,
                                                                          tag2idx)
    test_data = TensorDataset(torch.tensor(test_windows_idx), torch.tensor(test_window_tags_idx))
    test_dataloader = DataLoader(test_data, batch_size=128)

    model = Tagger3(vocab_size, hidden_dim, output_dim)
    model.load_state_dict(torch.load(f'model_part1_{TASK}.pth'))
    model.predict(test_dataloader, idx2tag, f'Output/part1.{TASK}')
