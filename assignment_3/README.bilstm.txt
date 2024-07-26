# BiLSTM Tagger

This repository contains a BiLSTM tagger for tasks such as POS tagging and NER (Named Entity Recognition). The model is built using LSTM (Long Short-Term Memory) networks and can be configured using various command-line arguments. Below is an explanation of each argument that can be used with the tagger.

## Arguments

### Positional Arguments
1. `repr` (str)
   - **Description:** The type of embeddings to use.
   - **Options:** `a`, `b`, `c`, or `d`
   - **Default:** `b`
   - **Usage:** This specifies which embedding representation the model will use.

2. `trainFile` (str)
   - **Description:** The path to the training samples file.
   - **Usage:** This file contains the data that the model will be trained on.

3. `modelFile` (str)
   - **Description:** The path to the file where the model will be saved.
   - **Usage:** After training, the model will be stored in this file.

### Optional Arguments
1. `--task` (str)
   - **Description:** The task to perform.
   - **Options:** `pos`, `ner`
   - **Default:** `ner`
   - **Usage:** Specifies whether the model should perform POS tagging or NER.

2. `--save_model` (str)
   - **Description:** Whether to save the model after training.
   - **Options:** `t` (true), `f` (false)
   - **Default:** `t`
   - **Usage:** If set to `t`, the trained model will be saved to `modelFile`. If `f`, the model will not be saved.

3. `--hidden_size` (int)
   - **Description:** The number of hidden units in the LSTM.
   - **Default:** `32`
   - **Usage:** Determines the size of the LSTM's hidden layer.

4. `--embedding_dim` (int)
   - **Description:** The size of the word embeddings.
   - **Default:** `32`
   - **Usage:** Specifies the dimensionality of the word embeddings used in the model.

5. `--char_hidden_size` (int)
   - **Description:** The size of the character LSTM hidden units.
   - **Default:** `4`
   - **Usage:** Determines the size of the hidden layer for the character-level LSTM.

6. `--epochs` (int)
   - **Description:** The number of training epochs.
   - **Default:** `5`
   - **Usage:** Specifies how many times the model will be trained on the entire dataset.

7. `--lr` (float)
   - **Description:** The learning rate for the optimizer.
   - **Default:** `0.001`
   - **Usage:** Determines the step size during gradient descent optimization.

## Example Usage

```bash
python tagger.py repr trainFile modelFile --task pos --save_model t --hidden_size 64 --embedding_dim 128 --char_hidden_size 8 --epochs 10 --lr 0.005
```

## Required file archtitecure
```bash
assignment_3
├── Code
│   ├── bilstmPredict.py
│   ├── bilstmTrain.py
│   └── utils.py
└── Data
    ├── ner
    │   ├── dev
    │   ├── test
    │   └── train
    ├── pos
    │   ├── dev
    │   ├── test
    │   └── train
    └── vocab.txt

```
    