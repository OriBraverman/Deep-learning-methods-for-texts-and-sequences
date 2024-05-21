import loglinear as ll
import train_loglin as classifier
from utils import *


def predict_test(test_data, model):
    preds = []
    for _, features in test_data:
        x = classifier.feats_to_vec(features)
        y_pred = ll.predict(x, model)
        y_pred_label = I2L[y_pred]
        preds.append(y_pred_label)
    return preds


def write_predictions(preds, filename):
    with open(filename, "w") as file:
        for pred in preds:
            file.write(str(pred) + "\n")



if __name__ == '__main__':
    train_data = [(l, t) for l, t in read_data("train")]
    dev_data = [(l, t) for l, t in read_data("dev")]
    num_iterations = 100
    learning_rate = 0.001
    in_dim = len(vocab)
    out_dim = len(L2I)
    params = ll.create_classifier(in_dim, out_dim)
    model = classifier.train_classifier(train_data, dev_data, num_iterations, learning_rate, params)
    test_data = [(l, t) for l, t in read_data("test")]
    predictions = predict_test(test_data, model)
    write_predictions(predictions, "test.pred")