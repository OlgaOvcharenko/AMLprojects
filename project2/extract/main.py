import optuna
import pandas as pd
import numpy as np
from extract import Extractor
from preprocess import preprocess
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import lightgbm as lgb

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks


def read_data(X_train_path, y_train_path, X_test_path, extract_data):  
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).iloc[:,1]

    if extract_data:
        train_ids, test_ids = X_train.iloc[:, 0], X_test.iloc[:, 0]
        X_train, X_test = X_train.iloc[:,1:], X_test.iloc[:,1:]
    else:
        train_ids, test_ids = pd.DataFrame(list(range(0, X_train.shape[0]))), \
            pd.DataFrame(list(range(0, X_test.shape[0])))

    return X_train, y_train, train_ids, X_test, test_ids


def get_splits(X_train: np.array, y_train: np.array, nfolds: int = 10):
    kf = StratifiedKFold(n_splits=nfolds, random_state=42, shuffle=True)
    return kf.split(X_train, y_train)


def get_model(method: int = 3):
    if method == 1:
        model = XGBClassifier()

    elif method == 2:
        model = CatBoostClassifier(iterations=1000, learning_rate=0.01, logging_level='Silent')

    elif method == 3:
        estimators = [ 
            ('cb', CatBoostClassifier(iterations=1000, learning_rate=0.01, logging_level='Silent')),
            # ('cb', CatBoostClassifier(iterations=1000, learning_rate= 0.0011611247781093898, depth=1, min_data_in_leaf=62, logging_level='Silent')),
            ('xgb', XGBClassifier(random_state=42)),
            # ('rf', RandomForestClassifier(n_estimators=200))
        ]
    
        model = StackingClassifier(estimators=estimators, final_estimator=CatBoostClassifier(iterations=1000, learning_rate=0.01, logging_level='Silent'))
    
    return model


def objective(trial):
    extract_data = False
    oversample = False

    # read data
    if extract_data:
        X_train_path, y_train_path, X_test_path = "data/X_train.csv", "data/y_train.csv", "data/X_test.csv"

    else:
        # X_train_path, y_train_path, X_test_path = "data/train_feat_new.csv", "data/y_train.csv", "data/test_feat_new.csv"
        # X_train_path, y_train_path, X_test_path = "data/train_combined.csv", "data/y_train.csv", "data/test_combined.csv"
        X_train_path, y_train_path, X_test_path = "data/train_combined.csv", "data/y_train.csv", "data/test_combined.csv"

    X_train, y_train, train_ids, X_test, test_ids = read_data(X_train_path, y_train_path, X_test_path, extract_data)

    print("Extracted / read data.")

    X_train, y_train, X_test = preprocess(X_train, y_train, X_test, drop_r=False)

    print("Preprocessed.")

    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)


    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "depth": trial.suggest_int("depth", 1, 10),
        # "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = CatBoostClassifier(**params, silent=True)
    model.fit(X_train_tune, y_train_tune)
    predictions = model.predict(X_val_tune)
    f1 = f1_score(y_val_tune, predictions, average='micro')
    return f1


def main():
    extract_data = False
    oversample = False

    # read data
    if extract_data:
        X_train_path, y_train_path, X_test_path = "data/X_train.csv", "data/y_train.csv", "data/X_test.csv"

    else:
        # X_train_path, y_train_path, X_test_path = "data/train_feat_new.csv", "data/y_train.csv", "data/test_feat_new.csv"
        # X_train_path, y_train_path, X_test_path = "data/train_combined.csv", "data/y_train.csv", "data/test_combined.csv"
        # X_train_path, y_train_path, X_test_path = "data/train_combined_lstm.csv", "data/y_train.csv", "data/test_combined_lstm.csv"
        X_train_path, y_train_path, X_test_path = "data/train_combined_cnn_64.csv", "data/y_train.csv", "data/test_combined_cnn_64.csv"

    X_train, y_train, train_ids, X_test, test_ids = read_data(X_train_path, y_train_path, X_test_path, extract_data)

    # extract
    if extract_data:
        extr = Extractor(X_train)
        train_feat = extr.extract()
        X_train = train_feat
        train_feat.to_csv("data/train_feat_new_old_dirty.csv", index=False)

        extr = Extractor(X_test)
        test_feat = extr.extract()
        X_test = test_feat
        test_feat.to_csv("data/test_feat_new_old_dirty.csv", index=False)
    
    print("Extracted / read data.")

    X_train, y_train, X_test = preprocess(X_train, y_train, X_test, drop_r=False)

    print("Preprocessed.")
    
    nfolds = 5
    splits = get_splits(X_train, y_train, nfolds)

    f1_scores = 0
    models = []
    for i, (train_index, test_index) in enumerate(splits):
        train_x, train_y, test_x, test_y = X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]

        if oversample:
            oversampler=SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
            train_x, train_y = oversampler.fit_resample(train_x, train_y)

            print("Oversample data.")

        model = get_model()
        model.fit(train_x, train_y)

        pred = model.predict(test_x)

        score = f1_score(test_y, pred, average="micro")

        print(f"Fold {i}: score {score}")
        f1_scores += score

        print(classification_report(test_y, pred))

        models.append(model)

    print(f"Avg F1: {f1_scores / nfolds}")

    if oversample:
        oversampler = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        print("Oversample data.")

    model_full = get_model()
    model_full.fit(X_train,y_train)
    res = model_full.predict(X_test)

    out = pd.DataFrame()
    out["id"] = test_ids
    out["y"] = res

    print(out.shape)
    print(out["id"].nunique())

    out.to_csv("data/out.csv", index=False)


def tune_params():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print('Best hyperparameters:', study.best_params)
    print('Best F1:', study.best_value)
    

main()
# tune_params()
