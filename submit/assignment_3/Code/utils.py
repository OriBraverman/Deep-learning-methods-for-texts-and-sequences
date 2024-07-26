"""
'utils.py' file implements the utility functions used in the assignment.
"""
import argparse
import copy
import inspect
import os
import sys
import random
from datetime import time

import tqdm
import torch
# import numpy as np
import logging as log
import torch.nn.functional as F

from torch import nn
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Any, Iterator
from pathlib import Path
from typing import NamedTuple, List

PAD_LETTER_TAG = '<PAD_LETTER>'

T2I_KEY = 't2i'

I2T_KEY = 'i2t'

I2W_KEY = 'i2w'

W2I_KEY = 'w2i'

UNKNOWN_TAG = '<UNK>'

PAD_TAG = '<PAD>'

# Constants
SEED = 1
GLOBAL_RANDOM_GENERATOR = torch.Generator()
SEPERATOR_MAP = {
    'POS': '\t',
    'NER': ' ',
    'POS_NEG': '\t'
}


class DatasetType:
    """
    @brief: Dataset types.
    """
    POS = 'POS'
    NER = 'NER'
    POS_NEG = 'POS_NEG'


class LSTM(nn.Module):
    """
    @brief: LSTM layer using LSTMCell.
    """

    def __init__(self, input_size, hidden_size, device, padding_idx, is_acceptor=True):
        """
        @brief: Initialize the LSTM layer.
        @param input_size: The size of the input vectors.
        @param hidden_size: The size of the hidden state.
        @param device: Device for computation.
        @param padding_idx: The index of the padding token.
        @param is_acceptor: True --> Accepter, False --> Transducer.
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.is_acceptor = is_acceptor
        self.lstm_cell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.padding_idx = padding_idx
        self.forward = self.acceptor_forward if is_acceptor else self.transducer_forward

    def init_hidden(self, batch_size):
        """
        @brief: Initialize the hidden state.
        @param batch_size: Batch size.
        """
        self.hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.cell = torch.zeros(batch_size, self.hidden_size).to(self.device)

    def acceptor_forward(self, sequence, original_lengths):
        """
        @brief: Forward pass for the LSTM acceptor.
        @param sequence: Input sequence.
        @param original_lengths: Original lengths of the input sequence.
        @return: The last non-padded hidden state.

        example:
        sequence = torch.tensor([[1, 2, 3], [4, <pad>, <pad>], [5, 6, <pad>]])
        original_lengths = torch.tensor([3, 1, 2])
        """
        batch_size, max_length, _ = sequence.size()

        # Initialize the hidden state
        self.init_hidden(batch_size)
        last_hidden = torch.zeros(batch_size, self.hidden_size).to(self.device)

        for t in range(max_length):
            # Get the current input
            x = sequence[:, t, :]

            # Update the hidden state
            self.hidden, self.cell = self.lstm_cell(input=x, hx=(self.hidden, self.cell))

            # Update the last hidden state
            last_hidden = torch.where((original_lengths > t).unsqueeze(1), self.hidden, last_hidden)

        return last_hidden

    def transducer_forward(self, sequence, original_length):
        """
        @brief: Forward pass for the LSTM transducer.
        @param sequence: Input sequence.
        @param original_lengths: Original lengths of the input sequence.
        @return: The last non-padded hidden states.

        example:
        sequence = torch.tensor([[1, 2, 3], [4, <pad>, <pad>], [5, 6, <pad>]])
        original_lengths = torch.tensor([3, 1, 2])
        """
        batch_size, max_length, _ = sequence.size()

        # Initialize the hidden state
        self.init_hidden(batch_size)
        hidden_states = []

        max_len = sequence != self.padding_idx


        for t in range(max_length):
            # Get the current input
            x = sequence[:, t, :]

            # Update the hidden state
            self.hidden, self.cell = self.lstm_cell(input=x, hx=(self.hidden, self.cell))
            hidden_states.append(self.hidden)

        return torch.stack(hidden_states, dim=1)


class BiLSTM(nn.Module):
    """
    @brief: Bi-directional LSTM layer using LSTMCell.
    """

    def __init__(self, input_size, hidden_size, device, padding_idx, original_len, is_acceptor=True):
        """
        @brief: Initialize the Bi-LSTM layer.
        @param input_size: The size of the input vectors.
        @param hidden_size: The size of the hidden state.
        @param device: Device for computation.
        @param padding_idx: The index of the padding token.
        @param is_acceptor: True --> Accepter, False --> Transducer.
        """
        super(BiLSTM, self).__init__()
        self.original_len = original_len
        self.hidden_size = hidden_size
        self.device = device
        self.padding_idx = padding_idx
        self.lstm_cell_forward = LSTM(input_size=input_size, hidden_size=hidden_size, device=device,
                                      padding_idx=padding_idx, is_acceptor=is_acceptor)
        self.lstm_cell_backward = LSTM(input_size=input_size, hidden_size=hidden_size, device=device,
                                       padding_idx=padding_idx, is_acceptor=is_acceptor)

    def forward(self, sequence):
        """
            @brief: Forward pass for the Bi-LSTM.
            @param sequence: Input sequence.
            @param original_lengths: Original lengths of the input sequence.
            @return: The concatenated hidden states from forward and backward LSTMs.
            """
        hidden_states_forward = self.lstm_cell_forward(sequence, self.original_len)
        hidden_states_backward = self.lstm_cell_backward(sequence.flip(dims=[1]), self.original_len)
        hidden_states_backward = hidden_states_backward.flip(dims=[1])

        hidden_states = torch.cat((hidden_states_forward, hidden_states_backward), dim=-1)
        return hidden_states


class BaseDataset(Dataset):
    """
    @brief: Base dataset class.
    """

    def __init__(self, device='', data_dir='', data_filename='', seperator=''):
        """
        @brief: Initialize the dataset.
        @param device: Device for computation.
        @param data_dir: Data directory.
        @param data_filename: Data filename.
        @param sep: Separator.
        """
        super().__init__()
        self.device = device
        self.data_dir = data_dir
        self.data_filename = data_filename
        self.seperator = seperator
        self.data_path = os.path.join(self.data_dir, self.data_filename)

        # Initialize placeholders for metadata
        self.metadata = {
            'vocab': set(),
            'vocab_tags': set(),
            'vocab_size': 0,
            'max_word_length': 0,
            'num_classes': 0,
            'word2idx': {},
            'idx2word': {},
            'tag2idx': {},
            'idx2tag': {},
            'unknown_token_idx': 0,
            'padding_token_idx': 0,
            'padding_tag_idx': 0
        }

    def __len__(self):
        """
        @brief: Get the length of the dataset.
        @return: Length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        @brief: Get an item from the dataset.
        @param idx: Index of the item.
        @return: Item from the dataset.
        """
        return self.X[idx], self.y[idx]

    def initialize(self, metadata=None):
        """
        @brief: Initialize the dataset.
        """
        self._initialize_metadata(metadata=metadata)
        self._to_indices()

    def get_metadata(self):
        """
        @brief: Get the metadata of the dataset.
        @return: Metadata of the dataset.
        """
        return self.metadata

    def _set_metadata(self, metadata):
        """
        @brief: Set the metadata of the dataset.
        @param metadata: Metadata of the dataset.
        """
        for key in metadata:
            if key in self.metadata:
                self.metadata[key] = metadata[key]

    def _initialize_metadata(self, metadata=None):
        """
        @brief: Initialize the metadata of the dataset.
        """
        if metadata:
            self._set_metadata(metadata)
        else:
            self._initialize_X()
            self._initialize_y()

    def _initialize_X(self):
        """
        @brief: Initialize the input sequences.
        """
        raise NotImplementedError("This method should be implemented in the child class.")

    def _initialize_y(self):
        """
        @brief: Initialize the target sequences.
        """
        raise NotImplementedError("This method should be implemented in the child class.")

    def _to_indices(self):
        """
        @brief: Convert the input and target sequences to indices.
        """
        raise NotImplementedError("This method should be implemented in the child class.")


class BinaryClassificationDataset(BaseDataset):
    """
    @brief: Binary classification dataset class.
    """

    def __init__(self, device='', data_dir='', data_filename='', seperator='\t'):
        """
        @brief: Initialize the dataset.
        @param device: Device for computation.
        @param data_dir: Data directory.
        @param data_filename: Data filename.
        @param sep: Separator.
        """
        super(BinaryClassificationDataset, self).__init__(device=device, data_dir=data_dir, data_filename=data_filename,
                                                          seperator=seperator)
        self._initialize_data()

    def _initialize_data(self):
        """
        @brief: Initialize the data.
        """
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.X, self.y = [], []
        for line in lines:
            x, y = line.strip().split(self.seperator)
            if x and y:
                self.X.append(x)
                self.y.append(y)

        assert len(self.X) == len(self.y), "Length of X and y should be the same."

        self.metadata['max_word_length'] = max([len(x) for x in self.X])

    def _initialize_X(self):
        """
        @brief: Initialize the input sequences.
        """
        vocab = set()
        for sentence in self.X:
            vocab.update(sentence)
        # Add padding and unknown tokens
        vocab.add('<pad>')
        vocab.add('<unk>')
        vocab_list = sorted(list(vocab))
        self.metadata['vocab'] = vocab
        self.metadata['vocab_size'] = len(vocab_list)
        self.metadata['word2idx'] = {word: idx for idx, word in enumerate(vocab_list)}
        self.metadata['idx2word'] = {idx: word for word, idx in self.metadata['word2idx'].items()}
        self.metadata['padding_token_idx'] = self.metadata['word2idx']['<pad>']
        self.metadata['unknown_token_idx'] = self.metadata['word2idx']['<unk>']

    def _initialize_y(self):
        """
        @brief: Initialize the target sequences.
        """
        self.metadata['vocab_tags'] = set(self.y)
        self.metadata['num_classes'] = len(self.metadata['vocab_tags'])
        self.metadata['tag2idx'] = {tag: idx for idx, tag in enumerate(self.metadata['vocab_tags'])}
        self.metadata['idx2tag'] = {idx: tag for tag, idx in self.metadata['tag2idx'].items()}

    def _to_indices(self):
        """
        @brief: Convert the input and target to tensors of indices (same dimension with padding).
        """
        self.X = [[self.metadata['word2idx'].get(char, self.metadata['unknown_token_idx']) for char in sentence] for
                  sentence in self.X]
        self.X = [torch.tensor(sentence) for sentence in self.X]
        self.X = nn.utils.rnn.pad_sequence(self.X, batch_first=True, padding_value=self.metadata['padding_token_idx'])

        self.y = [self.metadata['tag2idx'][tag] for tag in self.y]
        self.y = torch.tensor(self.y)


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and number of correct classifications.
    """
    loss: float
    num_correct: int


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    accuracy: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]


class TorchTrainer():
    """
    A class for training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, optimizer, scheduler, device):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param optimizer: The optimizer to train with.
        :param scheduler: The learning rate scheduler.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_val: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_val: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        best_loss = -1
        best_acc = -1
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement = \
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/val_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch + 1}/{num_epochs} ---', verbose)

            train_epoch_results = self.train_epoch(dl_train=dl_train)
            train_loss.extend(train_epoch_results.losses)
            train_acc.append(train_epoch_results.accuracy)

            val_epoch_results = self.test_epoch(dl_test=dl_val)
            val_loss.extend(val_epoch_results.losses)
            val_acc.append(val_epoch_results.accuracy)
            avg_loss = sum(val_epoch_results.losses) / len(val_epoch_results.losses)

            if self.scheduler:
                # self.scheduler.step(avg_loss)
                self.scheduler.step()
                self._print(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}', verbose)

            # Early Stopping
            if avg_loss < best_loss or best_loss == -1:
                best_loss = avg_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping:
                print("Early Stopping\n")
                break

            if val_epoch_results.accuracy > best_acc:
                best_acc = val_epoch_results.accuracy

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch + 1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_epoch_results, val_epoch_results, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, val_loss, val_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test/validation set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model.forward(X)
        loss = self.model.loss(outputs, y)  # output = loss(input, target)
        loss.backward()
        self.optimizer.step()
        num_correct = (outputs.argmax(dim=1) == y).sum().item()
        loss = loss.item()

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            loss = self.model.loss(outputs, y)
            num_correct = (outputs.argmax(dim=1) == y).sum().item()
            loss = loss.item()
        return BatchResult(loss, num_correct)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)


def random_split(dataset, dev_split):
    """
    @brief: Randomly split the dataset into train and dev sets.
    @param dataset: Dataset to split.
    @param dev_split: The ratio of the dev set.
    @return: Train and dev datasets.
    """
    dataset_size = len(dataset)
    dev_size = int(dev_split * dataset_size)
    train_size = dataset_size - dev_size

    temp_train_dataset, temp_dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size],
                                                                         generator=GLOBAL_RANDOM_GENERATOR)
    train_dataset, dev_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.X, train_dataset.y = [dataset.X[i] for i in temp_train_dataset.indices], [dataset.y[i] for i in
                                                                                            temp_train_dataset.indices]
    dev_dataset.X, dev_dataset.y = [dataset.X[i] for i in temp_dev_dataset.indices], [dataset.y[i] for i in
                                                                                      temp_dev_dataset.indices]

    return train_dataset, dev_dataset


def get_train_dev_data_loader(ds_type, data_filename, device, batch_size=1, shuffle=True, dev_ratio=0.1):
    """
    @brief: Get the train and dev data loaders.
    @param ds_type: Type of dataset.
    @param data_filename: Data filename.
    @param device: Device for computation.
    @param batch_size: Batch size.
    @param shuffle: Shuffle the data.
    @param dev_ratio: Ratio of the dev set.
    @return: Train and dev data loaders.
    """
    log.info(f'Loading {ds_type} data...')

    if ds_type == DatasetType.POS_NEG:
        dataset = BinaryClassificationDataset(device=device, data_filename=data_filename,
                                              seperator=SEPERATOR_MAP[ds_type])
    else:
        # yet to be implemented
        raise NotImplementedError(f'{ds_type} dataset is not implemented yet.')

    ds_train, ds_dev = random_split(dataset, dev_ratio)
    ds_train.initialize()
    ds_dev.initialize(metadata=ds_train.get_metadata())

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    dev_loader = DataLoader(ds_dev, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return train_loader, dev_loader


def get_test_data_loader(ds_type, data_filename, device, batch_size=1, metadata=None):
    """
    @brief: Get the test data loader.
    @param ds_type: Type of dataset.
    @param data_filename: Data filename.
    @param device: Device for computation.
    @param batch_size: Batch size.
    @param metadata: Metadata of the dataset.
    @return: Test data loader.
    """
    log.info(f'Loading {ds_type} data...')

    if ds_type == DatasetType.POS_NEG:
        dataset = BinaryClassificationDataset(device=device, data_filename=data_filename,
                                              seperator=SEPERATOR_MAP[ds_type])
    else:
        # yet to be implemented
        raise NotImplementedError(f'{ds_type} dataset is not implemented yet.')

    dataset.initialize(metadata=metadata)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return data_loader


def set_seed(seed):
    """
    @brief: Set the seed for reproducibility.
    @param seed: Seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    GLOBAL_RANDOM_GENERATOR.manual_seed(seed)
    # torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    @brief: Get the device (GPU or CPU) for computation.
    @return: Device.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    return device


def get_logger(log_file):
    """
    @brief: Get the logger for logging.
    @param log_file: Log file.
    @return: Logger.
    """
    log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[log.FileHandler(log_file), log.StreamHandler()])
    return log.getLogger()


def save_model(model, model_path):
    """
    @brief: Save the model.
    @param model: Model to save.
    @param model_path: Model path.
    """
    log.info(f"Saving model to: {model_path}")
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_parameters': model.parameters,
    }, model_path)
    log.info("Model saved successfully.")


def load_model(model, model_path, device):
    """
    @brief: Load the model.
    @param model: Model to load.
    @param model_path: Model path.
    @param device: Device for computation.
    """
    log.info(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.parameters = checkpoint['model_parameters']
    log.info("Model loaded successfully.")
    return model


def is_debugging():
    for frame in inspect.stack():
        if frame[1].endswith("pydevd.py"):
            return True
    return False


def evaluate(task_type, model, data_loader, output_file):
    """
    @brief: Predict the output using the model.
    @param task_type: Type of task.
    @param model: Model to use for prediction.
    @param data_loader: Data loader.
    @param output_file: Output file to save predictions.
    """
    log.info(f"Predicting {task_type} data...")
    model.eval()
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    with open(output_file, 'w') as f:
        for batch in data_loader:
            X, y = batch
            outputs = model(X)
            predictions = outputs.argmax(dim=1)
            for prediction in predictions:
                f.write(f'{prediction.item()}\n')
    log.info(f"Predictions saved to: {output_file}")


def read_all_words(vocab_path, files_dir):
    with open(vocab_path, 'r') as f:
        lines = f.readlines()

    lines.append(PAD_TAG + '\n')
    lines.append(UNKNOWN_TAG + '\n')

    word2idx = {w[:-1]: i for i, w in enumerate(lines)}
    idx2word = {i: w[:-1] for i, w in enumerate(lines)}

    for mode in ['train', 'dev', 'test']:
        file_path = os.path.join(files_dir, mode)
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:

            if not line == '\n':
                if mode == 'test':
                    w = line[:-1]

                else:
                    line.replace('\t', ' ')  # In POS there are spaces in ner there are '\t'
                    w, _ = line.split()

                if w not in word2idx and w.lower() not in word2idx:
                    idx = len(word2idx)
                    word2idx[w] = idx
                    idx2word[idx] = w

    return word2idx, idx2word


class TaggerDataset:
    def __init__(self, file_path, vocab_path, files_dir, word2idx, idx2word, test=False, max_len_train=False, mode='a',
                 tag2idx = None, idx2tag = None):

        self.file_train = file_path
        self.mode = mode

        self.word2idx, self.idx2word = word2idx, idx2word

        self.words, self.tags = self._read_file(test)
        if tag2idx is None:
            self.tag2idx, self.idx2tag = self._read_tags(self.tags)
        else:
            self.tag2idx, self.idx2tag = tag2idx, idx2tag

        self.n_tags = len(self.tag2idx)

        if not max_len_train:
            self.max_len = len(max(self.words, key=len))
        else:
            self.max_len = max_len_train

        self.data = []
        self.labels = []

        if self.mode == 'b':
            self._read_letters()

        if self.mode == 'c':
            self._read_pre_suf()

        for words, tags in zip(self.words, self.tags):
            X = []
            y = []
            for i in range(self.max_len):
                if i < len(words):

                    x = self.get_data_for_mode(words[i], self.mode)
                    X.append(x)
                    y.append(self.tag2idx[tags[i]] if not test else 0)
                else:
                    x = self.get_data_for_mode(" ", self.mode, is_padding=True)
                    X.append(x)
                    y.append(self.tag2idx[PAD_TAG])

            self.data.append(X)
            self.labels.append(y)

        self.seq_length = len(self.data[0])
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _read_pre_suf(self):
        self.idx_word2idx_pre = {}
        self.idx_word2idx_suf = {}
        for (word, idx) in self.word2idx.items():
            if not word == PAD_TAG:
                l = min(3, len(word))
                pre = word[:l]
                suf = word[len(word) - l:]

                if pre not in self.idx_word2idx_pre:
                    self.idx_word2idx_pre[idx] = len(self.idx_word2idx_pre)
                if suf not in self.idx_word2idx_suf:
                    self.idx_word2idx_suf[idx] = len(self.idx_word2idx_suf)
        self.idx_word2idx_pre[self.word2idx[PAD_TAG]] = len(self.idx_word2idx_pre)
        self.idx_word2idx_suf[self.word2idx[PAD_TAG]] = len(self.idx_word2idx_suf)

    def _read_file(self, test):
        words = []
        tags = []
        line_words = []
        line_tags = []

        with open(self.file_train, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line == '\n':
                if self.mode in ['a', 'c']:
                    words.append(line_words)
                    tags.append(line_tags)
                elif len(line_words) <= 70 and self.mode in ['b', 'd']:
                    words.append(line_words)
                    tags.append(line_tags)
                line_words = []
                line_tags = []
            else:
                if test:
                    w = line[:-1]
                    line_words.append(w)
                    line_tags.append('<TEST_TAG>')

                else:
                    line = line.replace('\t', ' ')  # In POS there are spaces in ner there are '\t'
                    w, t = line.split()

                    # For common nouns in the beginning of the sentence
                    if w not in self.word2idx:
                        w = w.lower()
                    line_words.append(w)
                    line_tags.append(t)
        return words, tags

    def _read_letters(self):
        letters = {}
        self.max_word_len = 0
        for word in self.word2idx:

            if len(word) > self.max_word_len:
                self.max_word_len = len(word)

            for letter in word:
                letters[letter] = 1

        self.letter2idx = {k: i for i, k in enumerate(letters.keys())}
        self.letter2idx[PAD_LETTER_TAG] = len(self.letter2idx)

    def _read_vocab(self, vocab_path):

        with open(vocab_path, 'r') as f:
            lines = f.readlines()

        lines.append(PAD_TAG + '\n')
        lines.append(UNKNOWN_TAG + '\n')

        word2idx = {w[:-1]: i for i, w in enumerate(lines)}
        idx2word = {i: w[:-1] for i, w in enumerate(lines)}

        return word2idx, idx2word

    def _read_tags(self, tags):
        all_tags = []
        for tag in tags:
            all_tags.extend(tag)
        all_tags = list(set(all_tags))
        all_tags.sort()
        all_tags.append(PAD_TAG)

        tag2idx = {t: i for i, t in enumerate(all_tags)}
        idx2tag = {i: t for i, t in enumerate(all_tags)}

        return tag2idx, idx2tag

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_dicts(self):
        return {
            W2I_KEY: self.word2idx,
            I2W_KEY: self.idx2word,
            I2T_KEY: self.idx2tag,
            T2I_KEY: self.tag2idx
        }

    def get_data_for_mode(self, word, mode, is_padding=False):

        if mode in ['a', 'c', 'd']:
            if is_padding:
                return self.word2idx[PAD_TAG]
            return self.word2idx[word] if word in self.word2idx else self.word2idx[word.lower()]

        elif mode == 'b':
            if is_padding:
                return [self.letter2idx[PAD_LETTER_TAG] for _ in range(self.max_word_len)]
            letter_vec = []
            for i in range(self.max_word_len):
                if i < len(word):
                    letter_vec.append(self.letter2idx[word[i]])
                else:
                    letter_vec.append(self.letter2idx[PAD_LETTER_TAG])
            return letter_vec


class TaggerBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, padding_idx, original_len, device='cpu',
                 ):
        super(TaggerBiLSTM, self).__init__()
        self.device =  torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.bilstm = BiLSTM(embedding_dim, hidden_size, self.device, padding_idx, original_len, is_acceptor=False)
        # TODO Try with different classifier arch
        self.classifier = nn.Linear(2 * hidden_size, output_size)  # 2 * hidden_size because of BiLSTM

    def forward(self, sequence):
        embedded = self.embedding(sequence)
        bilstm_out = self.bilstm(embedded)
        logits = self.classifier(bilstm_out)

        return torch.log_softmax(logits, dim=2)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_idx) -> torch.Tensor:
        """
        @brief: Calculate the loss.

        The loss is calculated without taking into account the padding label.

        @param y_pred: Output probabilities.
        @param y_true: True labels.

        @return: The loss.
        """
        batch_size, max_sequence_length, num_of_labels = y_pred.shape
        input = y_pred.view(batch_size * max_sequence_length, num_of_labels)
        target = y_true.contiguous().view(batch_size * max_sequence_length)
        loss = torch.functional.F.cross_entropy(
            input=input, target=target, reduction='none', ignore_index=pad_idx)
        mask = (target != pad_idx).view(-1).type(torch.FloatTensor)
        mask /= mask.shape[0]
        mask = mask.to(self.device)
        loss = loss.dot(mask) / mask.sum()

        return loss


class CharLSTM(nn.Module):
    def __init__(self, ab_size, embedding_dim, hidden_size, padding_idx, originial_len, device='cpu'):
        super(CharLSTM, self).__init__()
        self.padding_idx = padding_idx
        self.originial_len = originial_len
        self.embedding = nn.Embedding(ab_size, embedding_dim, padding_idx=padding_idx)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.lstm = LSTM(embedding_dim, hidden_size, device, padding_idx)

    def forward(self, sequence):
        output = []
        for s in range(sequence.shape[1]):
            x = sequence[:, s, :]
            embedded = self.embedding(x)
            original_lengths = (x != self.padding_idx).sum(dim=1)
            out = self.lstm.acceptor_forward(embedded, original_lengths)
            output.append(out)
        return torch.stack(output, dim=1)


class CharTaggerBiLSTM(nn.Module):
    def __init__(self, ab_size, embedding_dim, hidden_size, char_hidden_size, output_size, padding_idx,
                 char_padding_idx,
                 original_len, device='cpu'):
        super(CharTaggerBiLSTM, self).__init__()
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.embedding = CharLSTM(ab_size, embedding_dim, char_hidden_size, char_padding_idx, device)
        self.lstm = BiLSTM(char_hidden_size, hidden_size, device, padding_idx, original_len, is_acceptor=False)
        #self.classifier = nn.Linear(hidden_size *2, output_size)


        #self.load_state_dict(torch.load('../outputs/models/part3/model_b_ner_best.pth'))
        self.classifier_2 = nn.Sequential(nn.Linear(2 * hidden_size, 256, ), nn.ReLU(),
                                          nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, output_size))
        # 2 * hidden_size because of BiLSTM
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out = self.lstm(embedded)
        output = self.classifier_2(lstm_out)
        return torch.log_softmax(output, dim=2)

    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_idx) -> torch.Tensor:
        """
        @brief: Calculate the loss.

        The loss is calculated without taking into account the padding label.

        @param y_pred: Output probabilities.
        @param y_true: True labels.

        @return: The loss.
        """
        batch_size, max_sequence_length, num_of_labels = y_pred.shape
        input = y_pred.view(batch_size * max_sequence_length, num_of_labels)
        target = y_true.contiguous().view(batch_size * max_sequence_length)
        loss = torch.functional.F.cross_entropy(
            input=input, target=target, reduction='none', ignore_index=pad_idx)
        mask = (target != pad_idx).view(-1).type(torch.FloatTensor)
        mask /= mask.shape[0]
        if torch.backends.mps.is_available():
            mask = mask.to(torch.device('mps'))
        loss = loss.dot(mask) / mask.sum()

        return loss


class CBOWTagger(TaggerBiLSTM):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, padding_idx, original_len,
                 idx_word2idx_suf, idx_word2idx_pre, device='cpu'):
        super(CBOWTagger, self).__init__(vocab_size, embedding_dim, hidden_size, output_size, padding_idx, original_len,
                                         device)
        self.idx_word2idx_suf = idx_word2idx_suf
        self.idx_word2idx_pre = idx_word2idx_pre

        self.pre_embedding = nn.Embedding(len(idx_word2idx_pre), embedding_dim, padding_idx=len(idx_word2idx_pre) - 1)
        self.suf_embedding = nn.Embedding(len(idx_word2idx_suf), embedding_dim, padding_idx=len(idx_word2idx_suf) - 1)

    def forward(self, sequence: torch.Tensor):
        word_embedding = self.pre_embedding(sequence)

        pre_sequence = sequence.clone().cpu().apply_(self.idx_word2idx_pre.get)
        suf_sequence = sequence.clone().cpu().apply_(self.idx_word2idx_suf.get)

        pre_embedding = self.pre_embedding(pre_sequence.to(self.device))
        suf_embedding = self.suf_embedding(suf_sequence.to(self.device))

        embedded = word_embedding + pre_embedding + suf_embedding

        bilstm_out = self.bilstm(embedded)
        logits = self.classifier(bilstm_out)

        return torch.log_softmax(logits, dim=2)


class MaxLSTM(nn.Module):
    def __init__(self, vocab_size, ab_size, char_embedding_dim, embedding_dim, hidden_size, output_size, padding_idx,
                 char_padding_idx, original_len, device='cpu'):
        super(MaxLSTM, self).__init__()


        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        self.char_embedding = CharLSTM(ab_size, char_embedding_dim, embedding_dim, char_padding_idx, device)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.linear = nn.Linear(embedding_dim * 2, embedding_dim)

        self.bilstm = BiLSTM(embedding_dim, hidden_size, self.device, padding_idx, original_len, is_acceptor=False)
        self.classifier = nn.Linear(2 * hidden_size, output_size)  # 2 * hidden_size because of BiLSTM

    def forward(self, x_char, x_word):
        word_embedded = self.word_embedding(x_word.to(self.device))
        char_embedded = self.char_embedding(x_char.to(self.device))
        embedded = torch.cat((word_embedded, char_embedded), dim=2)
        embedded = F.relu(self.linear(embedded))
        bilstm_out = self.bilstm(embedded)
        logits = self.classifier(bilstm_out)

        return torch.log_softmax(logits, dim=2)


    def loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, pad_idx) -> torch.Tensor:
        """
        @brief: Calculate the loss.

        The loss is calculated without taking into account the padding label.

        @param y_pred: Output probabilities.
        @param y_true: True labels.

        @return: The loss.
        """
        batch_size, max_sequence_length, num_of_labels = y_pred.shape
        input = y_pred.view(batch_size * max_sequence_length, num_of_labels)
        target = y_true.contiguous().view(batch_size * max_sequence_length)
        loss = torch.functional.F.cross_entropy(
            input=input, target=target, reduction='none', ignore_index=pad_idx)
        mask = (target != pad_idx).view(-1).type(torch.FloatTensor)
        mask = mask.to(self.device)
        mask /= mask.shape[0]
        loss = loss.dot(mask) / mask.sum()

        return loss





