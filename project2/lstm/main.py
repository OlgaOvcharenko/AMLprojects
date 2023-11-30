import time

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import rnn as rnn
import classifiers as SVM

import lstm as lstm

num_features_avg = 0


def load_data(X_train: str, y_train: str, X_test: str, read_test: bool, read_train: bool, slice_ids: True):
    start_read = time.time()
    X_train = pd.read_csv(X_train).to_numpy() if read_train else None
    if slice_ids:
        X_train = X_train[:, 1:]

    X_test = pd.read_csv(X_test).to_numpy() if read_test else None

    X_test_ind = pd.DataFrame(list(range(0, X_test.shape[0])))
    if slice_ids:
        X_test_ind = X_test[:, 0] if read_test else None
        X_test = X_test[:, 1:] if read_test else None
    y_train = pd.read_csv(y_train).to_numpy()[:, 1:] if read_train else None

    # TODO fix later to K-Fold
    if read_train:
        rows = (X_train != 0).sum(1)
        global num_features_avg
        num_features_avg = int(rows.sum() / rows.shape[0])

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    else:
        X_train, X_val, y_train, y_val = None, None, None, None

    print(f"Reading data in {time.time() - start_read} seconds.")
    return X_train, y_train, X_val, y_val, X_test, X_test_ind


def replace_nan(data, slice_by_avg_len=False):
    start_nan = time.time()
    data[np.isnan(data)] = 0.0
    print(f"Remove nan data in {time.time() - start_nan} seconds.")
    print(data.shape)
    # Trial to cut
    if slice_by_avg_len:
        data = data[:, 0: num_features_avg]
        print(data.shape)
    return data


def preprocess_data(X_train, X_test, y_train):
    X_train = replace_nan(X_train, False)
    X_test = replace_nan(X_test, False)
    y_train = replace_nan(y_train, False)

    return X_train, y_train, X_test


def evaluate(y_val, y_pred):
    print(f"\nAccuracy score: {accuracy_score(y_val, y_pred)}")
    print(f"\nF1 score: {f1_score(y_val, y_pred, average='micro')}")
    print("\nPrediction report:")
    print(classification_report(y_val, y_pred))


def main_svm():
    # To train only or to create test only, model is saved
    read_train = True
    read_test = True

    # Temporal solution to validate
    # TODO later add cross-fold once model is set
    X_train, y_train, X_val, y_val, X_test, X_test_ind = load_data(X_train='Data/X_train.csv',
                                                                   y_train='Data/y_train.csv',
                                                                   X_test='Data/X_test.csv',
                                                                   read_train=read_train,
                                                                   read_test=read_test)

    # TODO not equal length of observations, how to handle tails
    if read_train:
        X_train = replace_nan(X_train, slice_by_avg_len=True)
        X_val = replace_nan(X_val, slice_by_avg_len=True)

    if read_test:
        X_test = replace_nan(X_test, slice_by_avg_len=True)

        model = SVM.SVM_model(model_name='voting')
        X_train, X_test, X_val = model.prepare_data(X_train, X_test, X_val)
        model.train(X_train, y_train)
        model.score(X_val, y_val)

        predictions = model.predict(X_test)

        # Save output
        res = {'id': X_test_ind, 'y': predictions}
        df = pd.DataFrame(res)
        df.to_csv('results/out_lstm.csv', index=False)


def main_rnn():
    # To train only or to create test only, model is saved
    read_train = True
    read_test = True

    # Temporal solution to validate
    # TODO later add cross-fold once model is set
    X_train, y_train, X_val, y_val, X_test, X_test_ind = load_data(X_train='./data/train_feat_new.csv',
                                                                   y_train='./data/y_train.csv',
                                                                   X_test='./data/test_feat_new.csv',
                                                                   read_train=read_train,
                                                                   read_test=read_test,
                                                                   slice_ids=False)
    print("Read data.")

    mm = StandardScaler()

    # TODO not equal length of observations, how to handle tails
    if read_train:
        print("Preprocessing.")
        X_train = replace_nan(X_train, slice_by_avg_len=True)
        X_val = replace_nan(X_val, slice_by_avg_len=True)

        X_train = mm.fit_transform(X_train)
        X_val = mm.transform(X_train)

        print("Training.")
        
        rnn.train_rnn(X_train, y_train)
        predictions = rnn.predict(X_val)

        evaluate(y_val=y_val, y_pred=predictions)

    if read_test:
        X_test = replace_nan(X_test, slice_by_avg_len=False)
        # X_test = mm.transform(X_test) if read_train else mm.fit_transform(X_test)

        predictions = rnn.predict(X_test)

        # Save output
        res = {'id': X_test_ind, 'y': predictions}
        print(len(X_test_ind))
        print(len(predictions))

        df = pd.DataFrame(res)
        df.to_csv('results/out_lstm.csv', index=False)


def main_lstm():
    # To train only or to create test only, model is saved
    read_train = True
    read_test = True

    # Temporal solution to validate
    # TODO later add cross-fold once model is set
    X_train, y_train, X_val, y_val, X_test, X_test_ind = load_data(
        # X_train='./project2/data/X_train.csv',
        # X_test='./project2/data/X_test.csv',
        # X_train='./project2/data/cleaned_X_train.csv',
        # y_train='./project2/data/y_train.csv',
        # X_test='./project2/data/cleaned_X_test.csv',

        X_train='./project2/data/train_combined.csv',
        y_train='./project2/data/y_train.csv',
        X_test='./project2/data/test_combined.csv',
        read_train=read_train,
        read_test=read_test,
        slice_ids=False)
    print("Read data.")

    mm = StandardScaler()

    # TODO not equal length of observations, how to handle tails
    if read_train:
        print("Preprocessing.")
        X_train = replace_nan(X_train, slice_by_avg_len=False)
        X_val = replace_nan(X_val, slice_by_avg_len=False)

        X_train = mm.fit_transform(X_train)
        X_val = mm.transform(X_val)

        print("Training.")
        
        lstm.train(X_train, y_train, X_val, y_val)
        predictions = lstm.predict(X_val)

        evaluate(y_val=y_val, y_pred=predictions)

    if read_test:
        X_test = replace_nan(X_test, slice_by_avg_len=False)
        X_test = mm.transform(X_test) if read_train else mm.fit_transform(X_test)

        predictions = lstm.predict(X_test)

        # Save output
        res = {'id': X_test_ind, 'y': predictions}
        print(len(X_test_ind))
        print(len(predictions))

        df = pd.DataFrame(res)
        df.to_csv('results/out_lstm.csv', index=False)


if __name__ == "__main__":
    main_lstm()