import cubist
import best
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
from preprocess import preprocess
from sklearn.metrics import r2_score
from skrvm import RVR

np.random.seed(42)


def read_data(X_train_path, y_train_path, X_test_path):
    X_train = np.genfromtxt(X_train_path, delimiter=",")
    y_train = np.genfromtxt(y_train_path, delimiter=",")
    X_test = np.genfromtxt(X_test_path, delimiter=",")
    return X_train, y_train, X_test


def get_model(method: int = 1):
    if method == 1:
        estimators = [
            # ('lr', RidgeCV()),
            # ('lasso', Lasso(alpha=0.134694)),
            # ('enet', ElasticNet(alpha=0.201, l1_ratio=0.005)),
            # ('lm', LinearRegression()),
            # ('kernel_ridge', KernelRidge(alpha=2.0, kernel='polynomial', degree=1, coef0=0.005)),
            ('xgb', XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, colsample_bytree=0.8)),
            ('extratree', ExtraTreesRegressor(n_estimators=1000, random_state=0)),
            ('adaboost', AdaBoostRegressor(n_estimators=1000, random_state=0)),
            # ('svr_lin', SVR(kernel='linear')),
            # ['svr_rbf', SVR(kernel='rbf')],
            ('mn', KNeighborsRegressor()),
        ]
        model = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))
    elif method == 2:
        estimators = [
            ('xgb', XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, colsample_bytree=0.8)),
            ('extratree', ExtraTreesRegressor(n_estimators=1000, random_state=0)),
            ('adaboost', AdaBoostRegressor(n_estimators=1000, random_state=0)),
            ('lgbm', LGBMRegressor()),
            ('svr_rbf', SVR(kernel='rbf')),
            ('mn', KNeighborsRegressor()),
            ('rvm', RVR(alpha=1e-06)),
            # ('cat', CatBoostRegressor()),
            ('gp', GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5), random_state=42)),
            ('cubist', cubist.Cubist()),
        ]
        model = StackingRegressor(estimators=estimators,
                                  final_estimator=RandomForestRegressor(n_estimators=100, random_state=42))

    return model


def get_splits(X_train: np.array, nfolds: int = 10):
    kf = KFold(n_splits=nfolds, random_state=42, shuffle=True)
    return kf.split(X_train)


def main():
    X_train, y_train, X_test = read_data(X_train_path="data/X_train.csv",
                                         y_train_path="data/y_train.csv",
                                         X_test_path="data/X_test.csv")
    ids_train, ids_test = X_train[1:, 0], X_test[1:, 0].astype(int)
    X_train, y_train, X_test = X_train[1:, 1:], y_train[1:, 1:].ravel(), X_test[1:, 1:]
    X_train, y_train, X_test = preprocess(X_train, y_train, X_test)

    print("Preprocessed.")

    nfolds = 10
    splits = get_splits(X_train, nfolds)

    print("\nModels and folds.")

    r2 = 0
    models = []
    for i, (train_index, test_index) in enumerate(splits):
        model = get_model()
        model.fit(X_train[train_index], y_train[train_index])
        pred = model.predict(X_train[test_index])
        score = r2_score(y_train[test_index], pred)
        r2 += score

        models.append(model)

        print(f"Fold {i} R2 score: {score}")

    print(f"\nAvg R2: {r2 / nfolds}")

    print("\nTrained.")

    from_folds = True
    if from_folds:
        pred = models[0].predict(X_test)
        for model in models[1:]:
            pred += model.predict(X_test)
        pred = pred / nfolds
    else:
        model = get_model()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    res = np.column_stack((ids_test, pred))
    np.savetxt("data/out.csv", res, fmt=['%1i', '%1.4f'], delimiter=",", header="id,y", comments='')


if __name__ == "__main__":
    main()
