"""
'utils.py' file implements the utility functions used in the assignment.
"""
import argparse
import os
import random
import torch
import numpy as np
import logging as log

# Constants
SEED = 1
GLOBAL_RANDOM_GENERATOR = torch.Generator()


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


def get_args():
    """
    @brief: Get the arguments for the assignment.
    @return: Arguments.
    """
    parser = argparse.ArgumentParser(description='Assignment 3')
    parser.add_argument('--train_file', type=str, default='data/train.txt', help='Training file')
    parser.add_argument('--test_file', type=str, default='data/test.txt', help='Testing file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--log_file', type=str, default='output/log.txt', help='Log file')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    parser.add_argument('--bidirectional', type=bool, default=False, help='Bidirectional')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=SEED, help='Seed')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Save interval')
    parser.add_argument('--load_model', type=str, default=None, help='Model to load')
    parser.add_argument('--save_model', type=str, default='output/model.pth', help='Model to save')
    return parser.parse_args()


def load_train_dataset(train_file):
    """
    @brief: Load the training dataset.
    @param train_file: Training file.
    @return: Training dataset.
    """
    with open(train_file, 'r') as f:
        lines = f.readlines()
    return lines
