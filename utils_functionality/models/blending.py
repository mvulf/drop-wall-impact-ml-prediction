from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from numpy import hstack


class BlendingModels:
    def __init__(self):
        self.lgbm = LGBMClassifier()
        self.xgb = XGBClassifier()
        self.models = [("lgbm", self.lgbm), ("xgb", self.xgb)]

    def fit(self, X_train, X_val, y_train, y_val):
        # fit all models on the training set and predict on hold out set
        meta_X = list()
        for _, model in self.models:
            # fit in training set
            model.fit(X_train, y_train)
            # predict on hold out set
            yhat = model.predict(X_val)
            yhat = yhat.reshape(len(yhat), 1)
            # store predictions as input for blending
            meta_X.append(yhat)
        # create 2d array from predictions, each set is an input feature
        meta_X = hstack(meta_X)
        # define blending model
        self.blender = CatBoostClassifier(verbose=False)
        # fit on predictions from base models
        self.blender.fit(meta_X, y_val)
        return self.blender

    def predict(self, X_test):
        meta_X = list()
        for _, model in self.models:
            yhat = model.predict(X_test)
            yhat = yhat.reshape(len(yhat), 1)
            # store predictions as input for blending
            meta_X.append(yhat)
        # create 2d array from predictions, each set is an input feature
        meta_X = hstack(meta_X)
        return self.blender.predict(X_test)
