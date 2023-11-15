import cubist
import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR, SVR
from xgboost import XGBRegressor
from preprocess import preprocess
from sklearn.metrics import r2_score, mean_squared_error
from skrvm import RVR

np.random.seed(42)


def read_data(X):
    return np.genfromtxt(X, delimiter=",")


def main():
    X1 = read_data(X="data_out/out_64.csv")
    X2 = read_data(X="data_out/out_best.csv")
    X3 = read_data(X="data_out/out_31.csv")
    # X4 = read_data(X="data_out/out_diff_classifier.csv")
    ids = X1[1:, 0]

    sum = 0.724507569893 + 0.736650544556 + 0.714748091135 # + 0.712364259615

    w1 = 0.724507569893 / sum
    w2 = 0.736650544556 / sum
    w3 = 0.714748091135 / sum
    w4 = 0.712364259615 / sum
    pred = (w1*X1[1:, 1] + w2*X2[1:, 1] + w3*X3[1:, 1])

    print(pred.shape)
    print(ids.shape)

    res = np.column_stack((ids, pred))
    np.savetxt("data_out/out_combined.csv", res, fmt=['%1i', '%1.4f'], delimiter=",", header="id,y", comments='')


if __name__ == "__main__":
    main()
