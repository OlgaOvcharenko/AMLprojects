import xgboost
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


class SVM_model:
    def __init__(self, model_name: str = 'hgb'):
        if model_name == 'stacking':
            estimators = [
                ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                ('svr', make_pipeline(StandardScaler(),
                                      LinearSVC(random_state=42))),
                ('hist', HistGradientBoostingClassifier())

            ]
            self.model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

        elif model_name == 'voting':
            xgb = xgboost.XGBClassifier(objective="multi:softprob", nthread=-1,
                                        subsample=0.9, reg_lambda=0.05, reg_alpha=0.7)
            gbrt = GradientBoostingClassifier(random_state=0)
            forest = RandomForestClassifier(n_jobs=-1, random_state=0)
            lr = LogisticRegression(C=0.03)

            self.model = VotingClassifier(estimators=[('xgboost', xgb), ('gbrt', gbrt),
                                                      ('forest', forest), ('logistic regression', lr)],
                                          voting='soft', weights=None)

        elif model_name == 'hgb':
            self.model = HistGradientBoostingClassifier()

        else:
            self.model = OneVsRestClassifier(LinearSVC(random_state=0))

    def prepare_data(self, X_train, X_test, X_val=None):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        pca = LocallyLinearEmbedding(n_components=2000)
        # pca = PCA(n_components=None)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        if X_val is not None:
            X_val = pca.transform(X_val)
        return X_train, X_test, X_val

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def score(self, X_val, y_val):
        y_pred = self.model.predict(X_val)
        print("\nAccuracy score:")
        print(accuracy_score(y_val, y_pred))
        print("\nPrediction report:")
        print(classification_report(y_val, y_pred))
