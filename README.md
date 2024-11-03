# Deep Learning Methods for Text and Sequences - Course Projects

This repository contains project implementations for the course **Deep Learning Methods for Texts and Sequences**, taught by Yoav Goldberg at Bar Ilan University. This course covers foundational methods in deep learning for handling textual and sequential data, with a focus on practical, hands-on implementations.

Throughout the course, we worked on a series of projects, each designed to deepen understanding of core deep learning techniques as they apply to language processing tasks. Each project is developed as a command-line application, with a structured library module for modularity and reusability.

### Project Overviews

- **Project 1 - Gradient-Based Classification**
  - This project implements two types of classifiers: a log-linear model and a multi-layer perceptron (MLP) with adjustable depth. Using only NumPy, both classifiers are implemented from the ground up. Models are trained with cross-entropy loss and evaluated on tasks like language identification and the XOR problem.

- **Project 2 - Window-Based Sequence Tagging**
  - In this project, a sequence tagger is developed using PyTorch. The model leverages window-based features and supports various word representations, including pre-trained word embeddings and sub-word units. This tagger is applied to multiple tagging tasks to explore performance across different representation techniques.

- **Project 3 - RNN Acceptors and BiRNN Transducers**
  - This project involves constructing and training RNN acceptors and exploring their ability to learn language patterns. Additionally, a bidirectional LSTM-based tagger is implemented for tasks like part-of-speech tagging and named entity recognition. This project features manual data batching and uses LSTM cells directly, without relying on high-level PyTorch LSTM modules.

Each project demonstrates progressively more complex neural network architectures and aims to provide hands-on experience with core methods for processing text and sequence data.
