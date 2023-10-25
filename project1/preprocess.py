import numpy as np
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import f_regression, SelectKBest


def preprocess(X_train: np.array, y_train: np.array, X_test: np.array):
    X_train, X_test = impute_mv(X_train, X_test)
    X_train, X_test = scale_data(X_train, X_test)

    # TODO
    X_train, X_test = detect_remove_outliers(X_train, X_test)

    X_train, X_test = select_features(X_train, y_train, X_test)
    return X_train, y_train, X_test


def select_features(X_train: np.array, y_train: np.array, X_test: np.array):
    # TODO try ANOVA

    fs = SelectKBest(score_func=f_regression, k=10)

    X_train = fs.fit_transform(X_train, y_train.ravel())
    X_test = fs.transform(X_test)
    return X_train, X_test


def scale_data(X_train: np.array, X_test: np.array, method: str = 'robust'):
    if method == 'robust':
        transformer = RobustScaler()
    elif method == 'min_max':
        transformer = MinMaxScaler()
    else:
        raise Exception(f"Scale: {method} is not implemented")

    transformer.fit_transform(X_train)
    transformer.transform(X_test)
    return X_train, X_test


def impute_mv(X_train: np.array, X_test: np.array, method: str = 'median'):
    if method == 'median':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')

    elif method == 'iterative':
        imp = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

    else:
        raise Exception(f"Impute: {method} is not implemented")

    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    return X_train, X_test


def detect_remove_outliers(X_train: np.array, X_test: np.array):
    # TODO
    # train_pred, test_pred = detect_outlier_obs(X_train, X_test)

    return X_train, X_test


def detect_outlier_obs(X_train: np.array, X_test: np.array, method: str = 'isolation_forest'):
    train_pred, test_pred = [], []
    if method == 'ECOD':
        for i in range(X_train.shape[1]):
            ecod = ECOD(contamination=0.05)
            ecod.fit(X_train[:, i].reshape(-1, 1))
            y_train_pred = ecod.labels_
            y_test_pred = ecod.predict(X_test[:, i].reshape(-1, 1))
            train_pred.append(y_train_pred)
            test_pred.append(y_test_pred)

    elif method == 'isolation_forest':
        for i in range(X_train.shape[1]):
            clf = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.1))
            y_train_pred = clf.fit_predict(X_train[:, i].reshape(-1, 1))
            y_test_pred = clf.predict(X_test[:, i].reshape(-1, 1))
            train_pred.append(y_train_pred)
            test_pred.append(y_test_pred)

            print(sum([t == -1 for t in y_test_pred]))

    else:
        raise Exception(f"Detect: {method} is not implemented")

    return train_pred, test_pred
