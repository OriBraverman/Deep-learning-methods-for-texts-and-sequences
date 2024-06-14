import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
# import the utils.py file from the father directory
from assignment_2.utils import *

# Constants
PART = 'Part_3'
TASK = 'pos'
TRAIN = True
TRAIN_BATCH_SIZE = 16 if TASK == 'pos' else 32
DEV_BATCH_SIZE = 16 if TASK == 'pos' else 32

class Tagger2(nn.Module):
    def __init__(self, vocab_size, embedding, hidden_dim, output_dim, embedding_dim=50, window_size=5):
        super(Tagger2, self).__init__()

        # Embedding layer - 50 dimensions
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding))
        # Fully connected
        self.fc1 = nn.Linear(embedding_dim * window_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)
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
            for words, tags in tqdm(train_data, desc=f'Epoch {epoch + 1}/{epochs}'):
                # Zero the gradients before the backward pass
                optimizer.zero_grad()
                # Move the data to the device
                words, tags = torch.tensor(words).to(device), torch.tensor(tags).to(device)
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

            # Early stopping
            if len(dev_loss_list) > 1 and dev_loss_list[-1] > dev_loss_list[-2]:
                print('Early stopping')
                break

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

    def predict(self, windows_idx, original_data, idx2tag, save_file, device='cpu'):

        # if file already exists, delete it
        if os.path.exists(save_file):
            os.remove(save_file)
        f = open(save_file, 'w')
        Predicted_tags = []
        with torch.no_grad():
            for words in windows_idx:
                words = torch.tensor(words).to(device).unsqueeze(0)
                output = self(words)
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


if __name__ == '__main__':
    vecs = np.loadtxt("Data/wordVectors.txt")

    with open("Data/vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]

    word2idx, idx2word = make_word_dict(vocab)

    """     TRAIN    """
    train_words, train_tags = read_data(f'Data/{TASK}/train')
    # Create the vocabularies
    _, _, tag2idx, idx2tag = make_vocabs(train_words, train_tags)

    vocab_size = len(word2idx)
    output_dim = len(tag2idx)
    hidden_dim = 128
    n_epoch = 25

    if TRAIN:
        # Create an output directory in which to save the generated files
        if not os.path.exists(f'{PART}/Output'):
            os.makedirs(f'{PART}/Output')

        # Convert the words to windows
        windows, window_tags = convert_words_to_window(train_words, train_tags, window_size=5)
        # Convert the windows to indices
        windows_idx, window_tags_idx = convert_window_to_window_idx(windows, window_tags, word2idx, tag2idx)

        # Cut the data to 1000 samples in order to debug faster
        #windows_idx = windows_idx[:1000]
        #window_tags_idx = window_tags_idx[:1000]


        model = Tagger2(vocab_size, vecs, hidden_dim, output_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_data = TensorDataset(torch.tensor(windows_idx), torch.tensor(window_tags_idx))
        train_dataloader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        # save the dataset
        #torch.save(train_data, f'dataset_tagger1_{TASK}.pth')

        print(len(train_data))
        dev_words, dev_tags = read_data(f'Data/{TASK}/dev')
        dev_windows, dev_window_tags = convert_words_to_window(dev_words, dev_tags, window_size=5)
        dev_windows_idx, dev_window_tags_idx = convert_window_to_window_idx(dev_windows, dev_window_tags, word2idx, tag2idx)
        dev_data = TensorDataset(torch.tensor(dev_windows_idx), torch.tensor(dev_window_tags_idx))
        dev_dataloader = DataLoader(dev_data, batch_size=DEV_BATCH_SIZE, shuffle=True)

        dev_loss_list, dev_accuracy_list = model.train(optimizer, train_dataloader, dev_dataloader, idx2tag, epochs=n_epoch,
                                                       is_ner=TASK == 'ner')

        make_graph(dev_loss_list, 'Loss over epochs', 'Loss', f'{PART}/Output/loss_{TASK}.png')
        make_graph(dev_accuracy_list, 'Accuracy over epochs', 'Accuracy', f'{PART}/Output/accuracy_{TASK}.png')

        torch.save(model.state_dict(), f'model_{PART}_{TASK}.pth')

    tag2idx['<TEST>'] = len(tag2idx)

    test_words = read_test_data(f'Data/{TASK}/test')
    test_windows = convert_words_to_window(test_words, window_size=5)
    test_windows_idx = convert_window_to_window_idx(test_windows, None, word2idx, tag2idx)

    model = Tagger2(vocab_size, vecs, hidden_dim, output_dim)
    model.load_state_dict(torch.load(f'model_{PART}_{TASK}.pth'))
    model.predict(test_windows_idx, test_words, idx2tag, f'{PART}/Output/test3.{TASK}')