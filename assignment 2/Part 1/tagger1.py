
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
# import the utils.py file from the father directory
from utils import *


class Tagger1(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, word2vec, embedding_dim=50, window_size=5):
        super(Tagger1, self).__init__()
        # Embedding layer - 50 dimensions
        #self.embedding = word2vec
        # Fully connected
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()
    def forward(self, x):
        #x = self.embedding(x)
        # Flatten the tensor to 1D
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def train(self, optimizer, loss_function, train_data, dev_data, idx2tag, epochs=10):
        dev_loss_list, dev_accuracy_list = [], []
        for epoch in range(epochs):
            total_loss = 0
            for data in train_data:
                # Zero the gradients before the backward pass
                optimizer.zero_grad()
                words, tags = data
                words = torch.tensor(words)
                tags = torch.tensor(tags)
                output = self(words)
                loss = loss_function(output, tags)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Calculate loss and accuracy on the training data
            total_loss /= len(train_data)
            train_accuracy, _ = self.evaluate(train_data, idx2tag)
            dev_accuracy, dev_loss = self.evaluate(dev_data, idx2tag)

            dev_loss_list.append(dev_loss)
            dev_accuracy_list.append(dev_accuracy)
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Dev Accuracy: {dev_accuracy:.4f}')


    def evaluate(self, data):
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for data in data:
                words, tags = data
                words = torch.tensor(words)
                tags = torch.tensor(tags)
                output = self(words)
                loss = self.loss_function(output, tags)
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += tags.size(0)
                correct += (predicted == tags).sum().item()
        return correct / total, total_loss / len(data)


if __name__ == '__main__':
    # Create an output directory in which to save the generated files
    if not os.path.exists('Output'):
        os.makedirs('Output')

    words, tags = read_data('Data/ner/train')
    word2idx, idx2word, tag2idx, idx2tag = make_vocabs(words, tags)
    vocab_size = len(word2idx)
    output_dim = len(tag2idx)
    hidden_dim = 100
    model = Tagger1(vocab_size, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train(optimizer, model.loss_function, train_data, dev_data, idx2tag, epochs=10)
    test_accuracy, test_loss = model.evaluate(test_data, idx2tag)
    print(f'Test Accuracy: {test_accuracy:.4f} - Test Loss: {test_loss:.4f}')
    torch.save(model.state_dict(), 'model1.pth')