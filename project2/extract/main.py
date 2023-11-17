import pandas as pd
import numpy as np
import pandas as pd
from extract import Extractor
from preprocess import preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
import lightgbm as lgb


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
            ('xgb', XGBClassifier(random_state=42)),
            # ('lgbm', lgb.LGBMClassifier(random_state=42))
        ]
    
        model = StackingClassifier(estimators=estimators, final_estimator=CatBoostClassifier(iterations=1000, learning_rate=0.01, logging_level='Silent'))
    
    return model


def main():
    extract_data = False

    # read data
    if extract_data:
        X_train_path, y_train_path, X_test_path = "data/X_train.csv", "data/y_train.csv", "data/X_test.csv"

    else:
        X_train_path, y_train_path, X_test_path = "data/train_feat.csv", "data/y_train.csv", "data/test_feat.csv"

    X_train, y_train, train_ids, X_test, test_ids = read_data(X_train_path, y_train_path, X_test_path, extract_data)

    # extract
    if extract_data:
        extr = Extractor(X_train)
        train_feat = extr.extract()
        X_train = train_feat
        train_feat.to_csv("data/train_feat.csv",index=False)

        extr = Extractor(X_test)
        test_feat = extr.extract()
        X_test = test_feat
        test_feat.to_csv("data/test_feat.csv",index=False)
    
    print("Extracted / read data.")

    X_train, y_train, X_test = preprocess(X_train, y_train, X_test, drop_r=False)

    print("Preprocessed.")
    
    nfolds = 5
    splits = get_splits(X_train, y_train, nfolds)

    model = get_model()
    f1_scores = 0
    for i, (train_index, test_index) in enumerate(splits):
        model = get_model()
        model.fit(X_train[train_index], y_train[train_index])

        pred = model.predict(X_train[test_index])

        score = f1_score(y_train[test_index], pred, average="micro")

        print(f"Fold {i}: score {score}")
        f1_scores += score

    print(f"Avg F1: {f1_scores / nfolds}")

    model_full = get_model()
    model_full.fit(X_train,y_train)
    res = model_full.predict(X_test)

    out = pd.DataFrame()
    out["id"] = test_ids.iloc[:, 0]
    out["y"] = res

    out.to_csv("data/out.csv", index=False)


main()
