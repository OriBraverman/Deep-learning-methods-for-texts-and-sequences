# Acceptor Tagger

## Overview
`experiment.py` is a script designed to train and test a Recurrent Neural Network (RNN) acceptor. The RNN used in this script is specifically a Long Short-Term Memory (LSTM) network followed by a Multi-Layer Perceptron (MLP) classifier with a single hidden layer. The network performs binary classification on input sequences.

## Language Definition
The script classifies sequences based on the following language definitions:

- **Positive Examples:** `[1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+`
- **Negative Examples:** `[1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+`

## Implementation Details
In this implementation, we used an LSTM cell to perform an LSTM layer for the tagger.

## Command Line Arguments

```plaintext
-d, --debug              Enable debugging (default: False)
--train                  Train the model (default: True)
--predict                Predict using the model (default: True)
--log                    Enable logging (default: True)
--log_dir                Directory to save logs (default: 'Logs')
--train_data_file        Training data file (default: 'Data/Pos_Neg_Examples/train')
--test_data_file         Test data file (default: 'Data/Pos_Neg_Examples/test')
--predict_output_file    Predictions output file (default: 'outputs/predictions/pos_neg_examples.pred')
--save_model_file        Model output file (default: 'outputs/models/pos_neg_examples.pth')
--load_model_file        Model input file (default: 'outputs/models/pos_neg_examples.pth')
--dev_ratio              Development ratio (default: 0.1)
--batch_size             Batch size (default: 32)
--epochs                 Number of epochs (default: 15)
--embedding_dim          Embedding dimension (default: 20)
--lstm_hidden_dim        LSTM hidden dimension (default: 32)
--mlp_hidden_dim         MLP hidden dimension (default: 16)
--lr                     Learning rate (default: 0.001)
--dropout                Dropout probability (default: 0.5)
--weight_decay           Weight decay (default: 1e-5)
--early_stopping         Early stopping (default: 6)
--seed                   Random seed (default: 42)
```

## Example Command
To run the script with the provided arguments:
```bash
python experiment.py --train True --predict True --train_data_file Data/Pos_Neg_Examples/train --test_data_file Data/Pos_Neg_Examples/test --predict_output_file outputs/predictions/pos_neg_examples.pred --save_model_file outputs/models/pos_neg_examples.pth --load_model_file outputs/models/pos_neg_examples.pth --dev_ratio 0.1 --batch_size 32 --epochs 15 --embedding_dim 20 --lstm_hidden_dim 32 --mlp_hidden_dim 16 --lr 0.001 --dropout 0.5 --weight_decay 1e-5 --early_stopping 6 --seed 42
```