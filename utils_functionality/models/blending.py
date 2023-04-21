from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from numpy import hstack


def get_models():
    models = list()
    models.append(('lgbm', LGBMClassifier()))
    models.append(('xgb', XGBClassifier()))
    return models


def fit_ensemble(models, X_train, X_val, y_train, y_val):
    # fit all models on the training set and predict on hold out set
    meta_X = list()
    for _, model in models:
        # fit in training set
        model.fit(X_train, y_train)
        # predict on hold out set
        yhat = model.predict_proba(X_val)
        # store predictions as input for blending
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # define blending model
    blender = CatBoostClassifier(verbose=False)
    # fit on predictions from base models
    blender.fit(meta_X, y_val)
    return blender

# make a prediction with the blending ensemble


def predict_ensemble(models, blender, X_test):
    # make predictions with base models
    meta_X = list()
    for _, model in models:
        # predict with base model
        yhat = model.predict_proba(X_test)
        # store prediction
        meta_X.append(yhat)
    # create 2d array from predictions, each set is an input feature
    meta_X = hstack(meta_X)
    # predict
    return blender.predict(meta_X)
