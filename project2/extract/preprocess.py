import numpy as np
import phate
import umap
from matplotlib import pyplot as plt
from mlxtend.plotting.pca_correlation_graph import corr2_coeff
from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.preprocessing import RobustScaler, MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import VarianceThreshold, RFE
from dataheroes import CoresetTreeServiceDTC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items
import seaborn as sns


def preprocess(X_train: np.array, y_train: np.array, X_test: np.array, drop_r: bool):
    if drop_r:
        X_train.drop([f"r{r}" for r in range(0, 154)], axis=1, inplace=True)
        X_test.drop([f"r{r}" for r in range(0, 154)], axis=1, inplace=True)
    # sns.heatmap(X_train.corr())

    X_train.dropna(axis=1, how='all', inplace=True)
    X_test = X_test[X_train.columns]

    X_train.replace(np.inf,np.nan,inplace=True)
    X_test.replace(np.inf,np.nan,inplace=True)

    X_train, X_test = impute_mv(X_train, X_test, 'median')
    # X_train, X_test = select_features(X_train, y_train, X_test)
    X_train, X_test = scale_data(X_train, X_test, 'standard')

    # X_train, y_train, X_test = detect_remove_outliers(X_train, y_train, X_test)

    return X_train, y_train, X_test


def make_polynomial(X_train: np.array, y_train: np.array, X_test: np.array, degree: int = 2):
    if degree > 2:
        raise Exception("make_polynomial: Insane degree.")

    poly = PolynomialFeatures(2)
    X_train = poly.fit_transform(X_train, y_train)
    X_test = poly.transform(X_test)

    print(X_train.shape)
    return X_train, X_test


def select_features(X_train: np.array, y_train: np.array, X_test: np.array):
    print(X_train.shape)
    X_train, X_test = remove_correlated(X_train, X_test)

    # fs = SelectKBest(score_func=f_regression, k=200)
    # X_train = fs.fit_transform(X_train, y_train.ravel())
    # X_test = fs.transform(X_test)
    # X_train, X_test = recursive_elemination(X_train, y_train, X_test)

    print(X_train.shape)
    return X_train, X_test


def recursive_elemination(X_train: np.array, y_train: np.array, X_test: np.array):
    model = LogisticRegression()
    rfe = RFE(model)

    X_train = rfe.fit_transform(X_train, y_train)
    X_test = rfe.transform(X_test)

    return X_train, X_test

def remove_correlated(X_train: np.array, X_test: np.array):
    # Constant features
    var_threshold = VarianceThreshold(threshold=0.0)  # TODO threshold = 0 for constant
    var_threshold.fit_transform(X_train)
    var_threshold.transform(X_test)

    # Correlated
    cor = corr2_coeff(X_train.T, X_train.T)
    p = np.argwhere(np.triu(np.isclose(cor, 1), 1))
    X_train = np.delete(X_train, p[:, 1], axis=1)
    X_test = np.delete(X_test, p[:, 1], axis=1)
    return X_train, X_test


def scale_data(X_train: np.array, X_test: np.array, method: str = 'min_max'):
    if method == 'robust':
        transformer = RobustScaler()
    elif method == 'min_max':
        transformer = MinMaxScaler()
    elif method == 'standard':
        transformer = StandardScaler()
    else:
        raise Exception(f"Scale: {method} is not implemented")

    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    return X_train, X_test


def impute_mv(X_train: np.array, X_test: np.array, method: str = 'iterative'):
    if method == 'median':
        imp = SimpleImputer(missing_values=np.nan, strategy='median')

    elif method == 'mean':
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')

    elif method == 'iterative':
        imp = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None, imputation_order='ascending')

    else:
        raise Exception(f"Impute: {method} is not implemented")

    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    return X_train, X_test


def detect_remove_outliers(X_train: np.array, y_train: np.array, X_test: np.array):
    # train_pred1 = detect_outlier_obs(X_train, y_train, 'coresets')
    train_pred2 = detect_outlier_obs(X_train, y_train, 'ECOD')
    # train_pred3 = detect_outlier_obs(X_train, X_test, 'elliptic')
    # train_pred4 = detect_outlier_obs(X_train, X_test, 'local_factor')

    train_pred = train_pred2

    print(X_train.shape)
    X_train = X_train[train_pred]
    print(X_train.shape)
    y_train = y_train[train_pred]

    return X_train, y_train, X_test


def detect_outlier_obs(X_train: np.array, y_train: np.array, method: str = 'elliptic'):
    train_pred, test_pred = [], []
    if method == 'ECOD':
        ecod = ECOD(contamination=0.03)
        ecod.fit(X_train)
        train_pred = np.array(ecod.labels_) == 0

    elif method == 'isolation_forest':
        for i in range(X_train.shape[1]):
            clf = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.03))
            y_train_pred = np.array(clf.fit_predict(X_train[:, i].reshape(-1, 1))) == 1
            train_pred.append(y_train_pred)
            # y_test_pred = np.array(clf.predict(X_test[:, i].reshape(-1, 1))) == 1
            # test_pred.append(y_test_pred)

        train_pred = np.array(train_pred).sum(axis=1)
        # test_pred = np.array(test_pred).sum(axis=1)

    elif method == "coresets":
        tree = CoresetTreeServiceDTC(optimized_for='cleaning')
        tree = tree.build(X=X_train, y=y_train, chunk_size=-1)
        result = tree.get_cleaning_samples(20)
        tree.remove_samples(result['idx'])
        res = tree.get_cleaning_samples(1212)
        # print(res['idx'].shape)
        # print(res["idx"])
        train_pred = np.full((X_train.shape[0],), False)
        train_pred[res["idx"]] = True

    elif method == "local_factor":
        lo = LocalOutlierFactor(n_neighbors=2)
        res = lo.fit_predict(X_train)
        train_pred = np.array(res) == 0

    elif method == "elliptic":
        ee = EllipticEnvelope(random_state=42)
        res = ee.fit_predict(X_train, y_train)
        train_pred = np.array(res) == 0

    else:
        raise Exception(f"Detect: {method} is not implemented")

    return train_pred.astype(int)
