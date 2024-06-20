"""
'utils.py' file implements the utility functions used in the assignment.
"""
import argparse
import copy
import inspect
import os
import sys
import random
import tqdm
import torch
import numpy as np
import logging as log

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Callable, Any
from pathlib import Path
from typing import NamedTuple, List



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
            last_hidden = torch.where(original_lengths > t, self.hidden, last_hidden)

        return last_hidden


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
        super(BinaryClassificationDataset, self).__init__(device=device, data_dir=data_dir, data_filename=data_filename, seperator=seperator)
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
        self.X = [[self.metadata['word2idx'].get(char, self.metadata['unknown_token_idx']) for char in sentence] for sentence in self.X]
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
            avg_loss = sum(train_epoch_results.losses) / len(train_epoch_results.losses)

            if self.scheduler:
                self.scheduler.step(avg_loss)
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

    temp_train_dataset, temp_dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size], generator=GLOBAL_RANDOM_GENERATOR)
    train_dataset, dev_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.X, train_dataset.y = [dataset.X[i] for i in temp_train_dataset.indices], [dataset.y[i] for i in temp_train_dataset.indices]
    dev_dataset.X, dev_dataset.y = [dataset.X[i] for i in temp_dev_dataset.indices], [dataset.y[i] for i in temp_dev_dataset.indices]

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
        dataset = BinaryClassificationDataset(device=device, data_filename=data_filename, seperator=SEPERATOR_MAP[ds_type])
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
        dataset = BinaryClassificationDataset(device=device, data_filename=data_filename, seperator=SEPERATOR_MAP[ds_type])
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    GLOBAL_RANDOM_GENERATOR.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
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