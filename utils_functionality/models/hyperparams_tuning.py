import numpy as np
import optuna

import catboost as cb
from sklearn.metrics import f1_score


def objective_cb(trial, train_x, valid_x, train_y, valid_y, param=None):
    if param is None:
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            )
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    gbm = cb.CatBoostClassifier(**param)

    gbm.fit(train_x, train_y, eval_set=[
            (valid_x, valid_y)], verbose=0, early_stopping_rounds=100)

    preds = gbm.predict(valid_x)
    pred_labels = np.rint(preds)
    f1 = f1_score(valid_y, pred_labels)
    return f1


def get_best_trial(train_x, valid_x, train_y, valid_y, n_trials=100, timeout=600, cb=True):
    study = optuna.create_study(direction="maximize")
    if cb:
        def obj(trial): return objective_cb(
            trial, train_x, valid_x, train_y, valid_y)
    study.optimize(obj, n_trials=n_trials, timeout=timeout)
    print("Number of finished trials: {}".format(len(study.trials)))
    trial = study.best_trial
    return trial
