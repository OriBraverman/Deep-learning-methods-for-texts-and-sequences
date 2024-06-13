import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from nltk.lm import Vocabulary
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Union
import os
import argparse
# import the utils.py file from the father directory
from assignment_2.utils import *

# Constants
TASK = 'pos'
TRAIN = True


class Tagger3(nn.Module):
    def __init__(self, vocab_size, pre_suf_size, hidden_dim, output_dim, embedding_dim=50, window_size=1,
                 use_prefix=True, use_suffix=True, use_word=True, save_pre_suff_embedding: Union[str, bool] = False,
                 aggregation = 'sum'):
        super(Tagger3, self).__init__()

        # Embedding layer - 50 dimensions
        word2vec = load_word2vec()
        word2vec = np.stack(list(word2vec.values()))

        self.use_word = int(use_word)
        self.use_prefix = int(use_prefix)
        self.use_suffix = int(use_suffix)
        self.save_pre_suff_embedding = save_pre_suff_embedding
        self.aggregation = aggregation

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))

        self.pre_embedding = nn.Embedding(pre_suf_size, embedding_dim)
        self.suf_embedding = nn.Embedding(pre_suf_size, embedding_dim)

        if not save_pre_suff_embedding:
            if os.path.exists('embeddings_prefix.npy'):
                self.pre_embedding.weight.data.copy_(torch.from_numpy(np.load('embeddings_prefix.npy')))
            if os.path.exists('embeddings_suffix.npy'):
                self.suf_embedding.weight.data.copy_(torch.from_numpy(np.load('embeddings_suffix.npy')))

        # Fully connected
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        if self.aggregation == 'sum':
            x = (self.embedding(x[:, 0]) * self.use_word
                 + self.pre_embedding(x[:, 1]) * self.use_prefix
                 + self.suf_embedding(x[:, 2]) * self.use_suffix)
        else:
            x = torch.cat([self.embedding(x[:, 0]) * self.use_word
                 , self.pre_embedding(x[:, 1]) * self.use_prefix
                 , self.suf_embedding(x[:, 2]) * self.use_suffix])

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

        if self.save_pre_suff_embedding:

            if self.save_pre_suff_embedding == 'prefix':
                emb = self.pre_embedding.weight.detach().numpy()
                np.save(f'embeddings_prefix.npy', emb)
            elif self.save_pre_suff_embedding == 'suffix':
                emb = self.suf_embedding.weight.detach().numpy()
                np.save(f'embeddings_suffix.npy', emb)
            else:
                emb = self.embedding.weight.detach().numpy()
                np.save(f'embeddings.npy', emb)

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

    def predict(self, data, original_data, idx2tag, save_file, device='cpu'):

        # if file already exists, delete it
        if os.path.exists(save_file):
            os.remove(save_file)
        f = open(save_file, 'w')
        with torch.no_grad():
            for (window_idx, _), original_sentence in zip(data, original_data):
                window_idx = torch.tensor(window_idx).to(device)
                output = self(window_idx)
                _, predicted = torch.max(output.data, 1)
                for w, p in zip(original_sentence, predicted):
                    f.write(f'{w} {idx2tag[p.item()]}\n')
                f.write('\n')
        f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pos')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--aggregation', type=str, default='sum')

    args = parser.parse_args()


    TASK = args.task
    TRAIN = args.train

    save_name = f'model_part4_{TASK}.pth'

    vocab = read_vocab()
    train_words, train_prefixes, train_suffixes, train_tags = read_data_pre_suf(f'Data/{TASK}/train')
    # Create the vocabularies
    word2idx, idx2word, tag2idx, idx2tag = make_vocabs_part4(vocab, train_tags)

    pre_suf_list = get_pre_suf_list(train_prefixes, train_suffixes)
    pre_suf2idx = make_pre_suf(pre_suf_list)

    vocab_size = len(word2idx)
    pre_suf_size = len(pre_suf2idx)
    output_dim = len(tag2idx)
    hidden_dim = args.hidden
    n_epoch = args.n_epochs
    batch_size = args.batch_size

    use_prefix = True
    use_suffix = True
    use_word = True
    save_emb = False

    if args.mode == 'prefix':
        use_prefix = True
        use_suffix = False
        use_word = False
        save_emb = 'prefix'

    if args.mode == 'suffix':
        use_prefix = False
        use_suffix = True
        use_word = False
        save_emb = 'suffix'

    if TRAIN:
        # Create an output directory in which to save the generated files
        if not os.path.exists('Output'):
            os.makedirs('Output')

        # Convert the words to windows
        windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=1)
        # Convert the windows to indices
        windows_idx, window_tags_idx = convert_window_to_window_idx_presuf(windows, window_tags, word2idx, pre_suf2idx,
                                                                           tag2idx)

        model = Tagger3(vocab_size, pre_suf_size, hidden_dim, output_dim,use_prefix=use_prefix, use_suffix=use_suffix,
                        use_word=use_word, save_pre_suff_embedding=save_emb)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # TODO SAVE THE DATASET TO SAVE TIME AT EACH RUN
        train_data = TensorDataset(torch.tensor(windows_idx), torch.tensor(window_tags_idx))
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        dev_words, dev_prefixes, dev_suffixes, dev_tags = read_data_pre_suf(f'Data/{TASK}/dev')

        dev_windows, dev_window_tags = convert_words_to_window(dev_words, dev_tags, window_size=1)
        # Convert the windows to indices
        dev_windows_idx, dev_window_tags_idx = convert_window_to_window_idx_presuf(dev_windows, dev_window_tags, word2idx,
                                                                                   pre_suf2idx,
                                                                                   tag2idx)
        dev_data = TensorDataset(torch.tensor(dev_windows_idx), torch.tensor(dev_window_tags_idx))
        dev_dataloader = DataLoader(dev_data, batch_size=batch_size, shuffle=True)

        dev_loss_list, dev_accuracy_list = model.train(optimizer, train_dataloader, dev_dataloader, idx2tag,
                                                       epochs=n_epoch,
                                                       is_ner=TASK == 'ner')

        make_graph(dev_loss_list, 'Loss over epochs', 'Loss', f'Output/loss_part4_{TASK}.png')
        make_graph(dev_accuracy_list, 'Accuracy over epochs', 'Accuracy', f'Output/accuracy_part4_{TASK}.png')



        torch.save(model.state_dict(), save_name)

    tag2idx['<TEST>'] = len(tag2idx)

    test_words,test_prefixes, test_suffixes, test_tags = read_test_data_pre_suf(f'Data/{TASK}/test')
    test_windows, test_window_tags = convert_words_to_window(test_words, test_tags, window_size=1)
    test_windows_idx, test_window_tags_idx = convert_window_to_window_idx_presuf(test_windows, test_window_tags, word2idx,
                                                                          pre_suf2idx, tag2idx)
    test_data = TensorDataset(torch.tensor(test_windows_idx), torch.tensor(test_window_tags_idx))
    test_dataloader = DataLoader(test_data, batch_size=128)

    model = Tagger3(vocab_size, pre_suf_size, hidden_dim, output_dim)
    model.load_state_dict(torch.load(save_name))
    model.predict(test_dataloader, test_words, idx2tag, f'Output/part4.{TASK}')
