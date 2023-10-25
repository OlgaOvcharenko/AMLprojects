import numpy as np
import umap
from mlxtend.plotting.pca_correlation_graph import corr2_coeff
from pyod.models.ecod import ECOD
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import f_regression, SelectKBest, chi2, VarianceThreshold


def preprocess(X_train: np.array, y_train: np.array, X_test: np.array):
    X_train, X_test = impute_mv(X_train, X_test)
    X_train, X_test = scale_data(X_train, X_test)

    # TODO
    X_train, X_test = detect_remove_outliers(X_train, X_test)
    X_train, X_test = select_features(X_train, y_train, X_test)

    X_train, X_test = reduce_dim(X_train, X_test)
    return X_train, y_train, X_test


def reduce_dim(X_train, X_test, method: str = 'PCA'):
    if method == 'PCA':
        reducer = PCA(n_components='mle', svd_solver='auto')

    elif method == 'UMAP':
        reducer = umap.UMAP()

    else:
        return X_train, X_test

    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)
    return X_train, X_test


def select_features(X_train: np.array, y_train: np.array, X_test: np.array):
    X_train, X_test = remove_correlated(X_train, X_test)

    # # Chi
    # f_p_values = chi2(X_train, y_train)
    # print(f_p_values)

    # Select k best
    fs = SelectKBest(score_func=f_regression, k=175)

    X_train = fs.fit_transform(X_train, y_train.ravel())
    X_test = fs.transform(X_test)
    return X_train, X_test


def remove_correlated(X_train: np.array, X_test: np.array):
    # Constant features
    var_threshold = VarianceThreshold(threshold=0)  # threshold = 0 for constant
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
    train_pred, test_pred = detect_outlier_obs(X_train, X_test)
    X_train = X_train[train_pred]
    X_test = X_test[train_pred]
    return X_train, X_test


def detect_outlier_obs(X_train: np.array, X_test: np.array, method: str = 'isolation_forest'):
    train_pred, test_pred = [], []
    if method == 'ECOD':
        for i in range(X_train.shape[1]):
            ecod = ECOD(contamination=0.05)
            ecod.fit(X_train[:, i].reshape(-1, 1))
            y_train_pred = np.array(ecod.labels_) == 0
            y_test_pred = np.array(ecod.predict(X_test[:, i].reshape(-1, 1))) == 0
            train_pred.append(y_train_pred)
            test_pred.append(y_test_pred)

    elif method == 'isolation_forest':
        for i in range(X_train.shape[1]):
            clf = IsolationForest(n_estimators=150, max_samples='auto', contamination=float(0.1))
            y_train_pred = np.array(clf.fit_predict(X_train[:, i].reshape(-1, 1))) == 1
            y_test_pred = np.array(clf.predict(X_test[:, i].reshape(-1, 1))) == 1
            train_pred.append(y_train_pred)
            test_pred.append(y_test_pred)

            print(sum([t == -1 for t in y_test_pred]))

    else:
        raise Exception(f"Detect: {method} is not implemented")

    return train_pred, test_pred
