import numpy as np
import optuna

import catboost as cb
from sklearn.metrics import f1_score
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def objective_cb(trial, train_x, valid_x, train_y, valid_y, cat_indexes, param=None):
    if param is None:
        param = {
            "objective": trial.suggest_categorical(
                "objective", ["Logloss", "CrossEntropy"]
            ),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]
            ),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10
            )
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param, cat_features=cat_indexes)

    gbm.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        verbose=0,
        early_stopping_rounds=100,
    )

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    f1 = f1_score(valid_y, pred_labels)
    return f1


def objective_rf(trial, train_x, valid_x, train_y, valid_y):
    params2 = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "n_estimators": trial.suggest_int("n_estimators", 50, 201),
        "max_features": trial.suggest_float("max_features", 0, 1),
        "max_depth": trial.suggest_int("max_depth", 1, 101),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 11),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 11),
    }
    rf = RandomForestClassifier(**params2)
    rf.fit(train_x, train_y)
    preds = rf.predict(valid_x)
    pred_labels = np.rint(preds)
    f1 = f1_score(valid_y, pred_labels)
    return f1


def objective_knn(trial, train_x, valid_x, train_y, valid_y):
    optimizer = trial.suggest_categorical(
        "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
    )
    n_neighbors = trial.suggest_int("n_neighbors", 1, 30)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical(
        "metric", ["euclidean", "manhattan", "minkowski"]
    )
    params2 = {
        "algorithm": optimizer,
        "n_neighbors": n_neighbors,
        "weights": weights,
        "metric": metric,
    }
    rf = KNeighborsClassifier(**params2)
    rf.fit(train_x, train_y)
    preds = rf.predict(valid_x)
    pred_labels = np.rint(preds)
    f1 = f1_score(valid_y, pred_labels)
    return f1


def get_best_trial(
    train_x,
    valid_x,
    train_y,
    valid_y,
    n_trials=100,
    timeout=600,
    cb=True,
    rf=False,
    knn=False,
    cat_indexes=None,
):
    study = optuna.create_study(direction="maximize")
    if cb:

        def obj(trial):
            return objective_cb(trial, train_x, valid_x, train_y, valid_y, cat_indexes)

    if rf:

        def obj(trial):
            return objective_rf(trial, train_x, valid_x, train_y, valid_y)

    if knn:

        def obj(trial):
            return objective_knn(trial, train_x, valid_x, train_y, valid_y)

    study.optimize(obj, n_trials=n_trials, timeout=timeout)
    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    return trial
