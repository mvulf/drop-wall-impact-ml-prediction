import logging

# Set logging level for PyTorch Tabular
logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)

# Set logging level for PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Set logging level for Lightning Fabric
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

# Disable device availability messages
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.FATAL)

import contextlib
import io


import pandas as pd
import numpy as np
import random

from pathlib import Path
import sys
import os
import joblib

import inspect

from tqdm.notebook import tqdm, trange

from collections.abc import Iterable
from functools import partial
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, KFold, ParameterGrid

import statsmodels.api as sm
from statsmodels.api import Logit

import torch  # Import PyTorch library for tensor computations
import torch.nn as nn  # Import neural network modules
import torch.nn.functional as F  # Import functional API for activation and loss functions
from torch.utils.data import DataLoader, TensorDataset, Dataset

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig
from torchmetrics import MeanMetric

import optuna

import copy

from IPython.display import display
import matplotlib.pyplot as plt

# from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append("../")
from utils_functionality.split_utils.split_tools import load_df, get_train_test

RANDOM_STATE = 42

# METRICS = ['f1_macro', 'roc_auc', 'balanced_accuracy']
METRICS = ['f1_score', 'accuracy']

class OptunaOptimizer:
    def __init__(
        self, 
        objective:callable,
        study_name:str,
        direction:str="maximize",
        seed:int=RANDOM_STATE,
    ):
        """Initialize the optimizer.

        Args:
            objective: The objective function to optimize.
            study_name: The name of the study.
            direction: The direction of optimization (maximize or minimize).
        """
        self.objective = objective
        self.study_name = study_name
        self.direction = direction
        
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(
                seed=seed,
            ),
        )
        
    def optimize(self, n_trials:int, **kwargs):
        """Optimize the objective function.

        Args:
            n_trials: The number of trials to run for optimization.

        Returns:
            The study object containing the optimization results.
        """
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            **kwargs,
        )


class GridSearchOptimizer:
    def __init__(
        self,
        objective:callable,
        param_grid:dict,
        verbose:bool=True,
    ):
        """
        Initialize the manual grid search optimizer.

        Args:
            objective: A callable like `objective(params_dict)` returning a score.
            param_grid: A dict of parameters to explore (like in sklearn).
        """
        self.objective = objective
        self.param_grid = list(ParameterGrid(param_grid))
        self.results = []
        self.best_score = -float('inf')
        self.best_params = None
        self.verbose = verbose
        
    def optimize(self):
        """Run the grid search over the parameter space."""
        
        for i, params in enumerate(
            tqdm(self.param_grid, desc="Optimizing parameters")
        ):
            score = self.objective(params)
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            self.results.append((score, params))
            if self.verbose:
                print(f"Progress: {i+1}/{len(self.param_grid)}.\tScore: {score}.\tConsidered params: {params}.")
        
        if self.verbose:
            print('-'*85)
            print(f"Best score: {self.best_score}")
            print(f"Best params: {self.best_params}")
            print('-'*85)

class MLPipeline:
    def __init__(
        self,
        *,
        target,
        estimator, # Estimator CLASS
        estimator_params:dict=None, # They would be passed to the estimator class
        base_estimator=None,
        base_estimator_params:dict=None,
        model_postfix="",
        dim_transformer=None,
        dim_transformer_params:dict=None,
        features_to_drop: tuple = (
            "Re",
            "We",
            "init_volume_fraction",
            "particle_liquid_density_ratio",
            "sedimentation_Re",
            "sign_particle_droplet_diameter_ratio",
            "sign_sedimentation_Re",
            "sign_sedimentation_Stk",
            # 'relative_roughness',
            # 'wettability',
            # 'inclination',
            # 'volume_fraction',
            # 'particle_droplet_diameter_ratio',
            # 'sedimentation_Stk',
            # 'K',
        ),
        minmax_features: tuple = None,
        passthrough_features: tuple = None,
        log_features: tuple = (
            "relative_roughness",
            "sedimentation_Re",
            "sedimentation_Stk",
            "particle_droplet_diameter_ratio",
        ),
        std_features=None,
        dataset_filename="df_dimless",
        path_data=Path("..", "data"),
        targets=("splashing", "no_fragmentation"),
        cv_folds=7,
        add_init_transformer=True,
        add_df_transformer=True,
        add_const=False,
        add_smote=True,
        is_smotenc=False,
        smote_params:dict=None,
        random_state=RANDOM_STATE,
        verbose=True,
        scoring_metrics=None,
        step_scoring_average:str="median",
        path_results=Path("..", "results"),
        models_folder="models_modelling4",
        metrics_file="metrics_modelling4.xlsx",
    ):
        # Copy estimator_params to init_estimator_params for update_estimator_params
        if not(estimator_params is None):
            init_estimator_params = estimator_params.copy()
        else:
            init_estimator_params = {}
            
        if not(base_estimator_params is None):
            init_base_estimator_params = base_estimator_params.copy()
        else:
            init_base_estimator_params = {}
        
        if not(dim_transformer_params is None):
            init_dim_transformer_params = dim_transformer_params.copy()
        else:
            init_dim_transformer_params = {}
            
        # Add features choice depending on the target
        if minmax_features is None:
            if target == "splashing":
                minmax_features = (
                    "inclination",
                    "init_volume_fraction",
                    "volume_fraction",
                )
            else:  # no_fragmentation
                minmax_features = ()
        if passthrough_features is None:
            if target == "splashing":
                passthrough_features = ("wettability",)
            else:  # no_fragmentation
                passthrough_features = (
                    "wettability",
                    "inclination",
                    "volume_fraction",
                    "init_volume_fraction",
                )

        target_set = set(targets)
        self._params = {
            "target": target,
            "dataset_filename": dataset_filename,
            "path_data": path_data,
            "target_set": target_set,
            "cv_folds": cv_folds,
            "step_scoring_average": step_scoring_average,
            "path_results": path_results,
            "models_folder": models_folder,
            "metrics_file": metrics_file,
        }
        if add_smote:
            smote_type = 'smote'
            if is_smotenc:
                smote_type += 'nc'
            model_postfix = '_'.join([smote_type, model_postfix])
        self.model_postfix = model_postfix
        self.verbose = verbose

        # Load dataframe
        self.full_df = load_df(
            dataset_filename=dataset_filename,
            path_data=path_data,
            target=target,
            target_set=target_set,
            verbose=self.verbose,
        )

        # Split train and test
        self.train, self.test = get_train_test(
            df=self.full_df,
            target=target,
            path_data=path_data,
            verbose=self.verbose,
        )

        # Get features
        source_features = list(self.full_df.columns)
        source_features.remove(target)
        source_features = tuple(source_features)
        # Prepare pipeline-params
        self._pipeline_params = {
            "estimator": estimator,
            "estimator_params": estimator_params,
            "init_estimator_params": init_estimator_params,
            "base_estimator": base_estimator,
            "base_estimator_params": base_estimator_params,
            "init_base_estimator_params": init_base_estimator_params,
            "dim_transformer": dim_transformer,
            "dim_transformer_params": dim_transformer_params,
            "init_dim_transformer_params": init_dim_transformer_params,
            "source_features": source_features,
            "features_to_drop": features_to_drop,
            "log_features": _drop_features(log_features, features_to_drop),
            "minmax_features": _drop_features(minmax_features, features_to_drop),
            "passthrough_features": _drop_features(
                passthrough_features, features_to_drop
            ),
            'std_features': std_features,
            'add_init_transformer': add_init_transformer,
            'add_df_transformer': add_df_transformer,
            'add_const': add_const,
            'add_smote': add_smote,
            'is_smotenc': is_smotenc,
            'smote_params': smote_params,
            'random_state': random_state,
        }
        # Prepare STD-features
        if std_features is None:
            features_to_drop_std = (
                features_to_drop
                + self._pipeline_params["minmax_features"]
                + self._pipeline_params["passthrough_features"]
            )
            self._pipeline_params["std_features"] = _drop_features(
                source_features, features_to_drop_std
            )
            if verbose:
                print("std_features")
                display(self._pipeline_params["std_features"])
        else:
            self._pipeline_params["std_features"] = _drop_features(
                std_features, features_to_drop
            )

        # Create full pipeline
        self.pipe = _create_pipeline(
            verbose=self.verbose, **self._pipeline_params
        )

        # Get pipeline name
        estimator_class_name = self.pipe.steps[-1][-1].__class__.__name__

        if estimator_class_name == "StatsModelsEstimator":
            estimator_class_name = "Logit"
            
        if estimator_class_name == 'PytorchTabularEstimator':
            estimator_class_name = self.pipe.steps[-1][-1].__name__
            # if estimator_class_name == 'CategoryEmbeddingModelConfig':
            #     estimator_class_name = 'CategoryEmbeddingModel'
            if 'Config' in estimator_class_name:
                estimator_class_name = estimator_class_name.replace(
                    'Config', ''
                )

        if estimator_class_name == "DecisionStumpEstimator":
            estimator_class_name = "DecisionStump"

        self.model_name = "_".join([estimator_class_name, self._params["target"]])
        if self.model_postfix:
            self.model_name = "_".join([self.model_name, self.model_postfix])

        if scoring_metrics is None:
            # NOTE: in new sklearn versions use response_method parameter instead of needs_proba
            # Metrics with pos_label=1 are equal to regular methods.
            self.scoring_metrics = {
                "accuracy": make_scorer(accuracy_score),
                "precision": make_scorer(precision_score),
                "recall": make_scorer(recall_score),
                "f1": make_scorer(f1_score),
                "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
                "f1_macro": make_scorer(f1_score, average="macro"),
                "accuracy_balanced": make_scorer(balanced_accuracy_score),
                "precision_class_0": make_scorer(precision_score, pos_label=0),
                "recall_class_0": make_scorer(recall_score, pos_label=0),
                "f1_class_0": make_scorer(f1_score, pos_label=0),
            }
        else:
            self.scoring_metrics = scoring_metrics

        self.step_metrics = [] # for step 
        self.metric_results = [] # for run
       
    def step_transformer(
        self,
        dim_transformer=None,
        dim_transformer_params:dict=None,
        shuffle:bool=True,
        verbose=False,
    ):
        self._pipeline_params['verbose'] = verbose
        dim_transformer_class = self._pipeline_params['dim_transformer']
        
        if dim_transformer is not None:
            self._pipeline_params['dim_transformer'] = dim_transformer_class
        if dim_transformer_params is not None:
            self._pipeline_params['dim_transformer_params'] = dim_transformer_params
            
        # Create full pipeline
        self.pipe = _create_pipeline(
            **self._pipeline_params
        )
        
        # Find step with VAE, or PCA or other dimentionality reduction transformer
        for i, step in enumerate(self.pipe.steps):
            if isinstance(step[1], dim_transformer_class):
                dim_transformer = self.pipe[i]
                pre_transformers = self.pipe[:i]
        
        if dim_transformer is None:
            raise ValueError(
                "No dimensionality reduction transformer found in the pipeline"
            )
        
        # y_train_full is used for SMOTE
        X_train_full, y_train_full = self.get_X_y(self.train) # train dataset
        
        n_splits = self._params["cv_folds"]
        cv = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=self._pipeline_params['random_state'],
        )
        
        metrics_train = np.zeros(n_splits)
        metrics_val = np.zeros(n_splits)
        
        if len(self.scoring_metrics) != 1:
            raise ValueError(
                f"Only one scoring metric is allowed, but received: {self.scoring_metrics}"
            )
        
        metric_name = list(self.scoring_metrics.keys())[0]
        metric = self.scoring_metrics[metric_name]

        for i, (train_idx, val_idx) in enumerate(cv.split(X_train_full)):
            if isinstance(X_train_full, pd.DataFrame):
                X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
            else:
                X_train, X_val = X_train_full[train_idx], X_train_full[val_idx]
            
            y_train, y_val = y_train_full[train_idx], y_train_full[val_idx]
            
            # X_train_preproc = pre_transformers.fit_transform(X_train)
            X_train_resampled, y_train_resampled = (
                pre_transformers.fit_resample(X_train, y_train)
            )
            
            # No need in SMOTE for validation set
            X_val_preproc = pre_transformers[:-1].transform(X_val)
            
            # dim_transformer.fit(X_train_resampled)
            dim_transformer.fit(X_train_resampled, y_train_resampled)
            X_train_reconstr = dim_transformer.inverse_transform(
                dim_transformer.transform(X_train_resampled)
            )
            X_val_reconstr = dim_transformer.inverse_transform(
                dim_transformer.transform(X_val_preproc)
            )
            
            # MSE by default
            metrics_train[i] = metric(X_train_resampled, X_train_reconstr) 
            metrics_val[i] = metric(X_val_preproc, X_val_reconstr)
        
        # Get average from CV metric on train set
        if self._params["step_scoring_average"] == "median":
            score_train = np.median(metrics_train)
            score_val = np.median(metrics_val)
        elif self._params["step_scoring_average"] == "mean":
            score_train = np.mean(metrics_train)
            score_val = np.mean(metrics_val)
        else:
            raise ValueError(f"Invalid step_scoring_average: {self._params['step_scoring_average']}")

        return score_val, score_train # score
    
        
    def step(
        self,
        estimator,
        estimator_params:dict=None,
        base_estimator=None,
        base_estimator_params:dict=None,
        dim_transformer=None,
        dim_transformer_params:dict=None,
        add_smote=None,
        is_smotenc=None,
        smote_params=None,
        verbose=False,
    ):
        # Update pipeline params (if provided)
        self._pipeline_params['estimator'] = estimator
        if estimator_params is not None:
            self._pipeline_params['estimator_params'] = estimator_params
        if add_smote is not None:
            self._pipeline_params['add_smote'] = add_smote
        if is_smotenc is not None:
            self._pipeline_params['is_smotenc'] = is_smotenc
        if smote_params is not None:
            self._pipeline_params['smote_params'] = smote_params
        self._pipeline_params['verbose'] = verbose
        
        if base_estimator is not None:
            self._pipeline_params['base_estimator'] = base_estimator
        if base_estimator_params is not None:
            self._pipeline_params['base_estimator_params'] = base_estimator_params
        if dim_transformer is not None:
            self._pipeline_params['dim_transformer'] = dim_transformer
        if dim_transformer_params is not None:
            self._pipeline_params['dim_transformer_params'] = dim_transformer_params
            
        # Create full pipeline
        self.pipe = _create_pipeline(
            **self._pipeline_params
        )
        
        X_train, y_train = self.get_X_y(self.train) # train dataset
        
        metrics = self.get_cv_metrics(
            X=X_train,
            y=y_train,
            cv_folds=self._params["cv_folds"],
            random_state=self._pipeline_params['random_state'],
            type="cv",
        )
        
        self.step_metrics.append(
            metrics
        )
        
        # Get average from CV metric on train set
        if self._params["step_scoring_average"] == "median":
            score = np.median(
                metrics['test_' + list(self.scoring_metrics.keys())[-1]]
            )
        elif self._params["step_scoring_average"] == "mean":
            score = np.mean(
                metrics['test_' + list(self.scoring_metrics.keys())[-1]]
            )
        else:
            raise ValueError(f"Invalid step_scoring_average: {self._params['step_scoring_average']}")

        return score # score

    def run(
        self,
        verbose=True,
        cv_verbose:bool|None=None,
        random_state=RANDOM_STATE,
        save_model_and_metrics=True,
    ):
        if cv_verbose is None:
            cv_verbose = verbose

        # Split X, y for fitting and predicting
        X_train, y_train = self.get_X_y(self.train) # train dataset
        X_test, y_test = self.get_X_y(self.test) # holdout test
        X, y = self.get_X_y(self.full_df) # full dataset

        # Conduct cross-validation
        metric_results_list = []
        metric_results_list.append(
            self.get_cv_metrics(
                X=X,
                y=y,
                cv_folds=self._params["cv_folds"],
                random_state=random_state,
                type="cv",
                verbose=cv_verbose,
            )
        )

        # Fit on holdout train dataset and get summary (if applicable)
        self.fit(
            X=X_train,
            y=y_train,
        )
        if verbose:
            self.get_summary()

        # Predict on train, test and save metrics
        metric_results_list.insert(
            0,
            self.get_metrics(
                X=X_test,
                y_true=y_test,
                type="holdout",
                verbose=False,
                prefix="test",
            ),
        )
        metric_results_list.insert(
            1,
            self.get_metrics(
                X=X_train,
                y_true=y_train,
                type="holdout",
                verbose=False,
                prefix="train",
            ),
        )

        # Transform metric_results_list to dict and append to metric_results
        metric_results_dict = {}
        for metrics in metric_results_list:
            type = metrics["type"]

            for key in metrics:
                if key != "type":
                    metric_results_dict["_".join((type, key))] = metrics[key]
        self.metric_results.append(metric_results_dict)

        # Prepare dataframe of final metrics
        df = pd.DataFrame(self.metric_results)

        df["dataset"] = self._params["dataset_filename"]
        df["target"] = self._params["target"]
        df["model"] = self.model_name
        
        # Replace estimator with its name
        params_dict =copy.deepcopy(self._pipeline_params)
        params_dict['estimator'] = params_dict['estimator'].__name__
        # If model_class is in params_dict['estimator_params'], 
        # replace model_class in estimator_params with its name
        if 'model_class' in params_dict['estimator_params']:
            params_dict['estimator_params']['model_class'] = (
                params_dict['estimator_params']['model_class'].__name__
            )
        # Remove init_//_params from params_dict
        params_dict.pop('init_estimator_params', None)
        params_dict.pop('init_base_estimator_params', None)
        params_dict.pop('init_dim_transformer_params', None)
        # Process base_estimator if it is used
        if params_dict['base_estimator'] is not None:
            params_dict['base_estimator'] = params_dict['base_estimator'].__name__
        else:
            params_dict.pop('base_estimator', None)
            params_dict.pop('base_estimator_params', None)
        # Process dim_transformer if it is used
        if params_dict['dim_transformer'] is not None:
            params_dict['dim_transformer'] = params_dict['dim_transformer'].__name__
        else:
            params_dict.pop('dim_transformer', None)
            params_dict.pop('dim_transformer_params', None)
        
        df["params"] = str(params_dict)

        df = pd.concat(
            (
                df.iloc[:, -4:],
                df.iloc[:, :-4],
            ),
            axis=1,
        )

        # ADD STATS
        cv_columns = [x for x in df.columns if x.startswith("cv_")]
        for cv_column in cv_columns:
            df[cv_column + "_std"] = df[cv_column].apply(lambda x: np.std(x))
            df[cv_column + "_mean"] = df[cv_column].apply(lambda x: np.mean(x))
            df[cv_column + "_median"] = df[cv_column].apply(lambda x: np.median(x))
        cv_columns = [x for x in df.columns if x.startswith("cv_")]
        non_cv_columns = [x for x in df.columns if not x.startswith("cv_")]
        df = df[non_cv_columns + sorted(cv_columns)]

        self.df_results = df.copy(deep=True)

        if verbose:
            display(
                self.df_results[
                    [
                        "target",
                        "model",
                        "holdout_test_f1_macro",
                        "holdout_test_accuracy_balanced",
                        "holdout_test_roc_auc",
                        "holdout_test_f1",
                        "holdout_test_accuracy",
                        "cv_test_f1_macro_median",
                        "cv_test_accuracy_balanced_median",
                        "cv_test_roc_auc_median",
                        "cv_test_f1_median",
                        "cv_test_accuracy_median",
                        "cv_test_precision_class_0_median",
                        "cv_test_recall_class_0_median",
                        "cv_test_f1_class_0_median",
                    ]
                ].T
            )

        # Save metrics and model
        if save_model_and_metrics:
            self.save_results(self.df_results)
            self.save_model()

    def list2str(self, value):
        # for value in col:
        if isinstance(value, np.ndarray):
            value = value.tolist()
            value = map(str, value)
            return ", ".join(value)
        return value

    def save_results(self, df_results):

        df_results_excel = df_results.applymap(self.list2str)

        filepath = Path(self._params["path_results"], self._params["metrics_file"])
        if os.path.isfile(filepath):
            existing_df = pd.read_excel(filepath)
            combined_df = pd.concat((existing_df, df_results_excel), ignore_index=True)

            # columns_to_check = [
            #     col for col in combined_df.columns if not('time' in col)
            # ]
            # columns_to_check = (
            #     set(combined_df.columns)
            #     - set(['cv_fit_time', 'cv_score_time'])
            # )
            columns_to_check = ["dataset", "target", "model", "params"]

            combined_df.drop_duplicates(
                subset=columns_to_check,
                keep="last",
                inplace=True,
            )
            combined_df.to_excel(filepath, index=False)
        else:
            df_results_excel.to_excel(filepath, index=False)

    def save_model(self, verbose=True):
        path_models = Path(
            self._params["path_results"],
            self._params["models_folder"],
        )
        if not os.path.exists(path_models):
            os.makedirs(path_models)
        model_path = path_models / self.model_name
        joblib.dump(self.pipe, model_path)
        if verbose:
            print(f"Model saved in {model_path}")

    def get_cv_metrics(
        self,
        *,
        X,
        y,
        cv_folds=7,
        random_state=None,
        shuffle=True,
        type:str="cv",
        verbose=True,
        fmt=".4f",
    ):
        df = X.copy()
        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=shuffle,
            random_state=random_state,
        )

        pipe = deepcopy(self.pipe)
        if not verbose:
            for step in pipe.steps:
                if 'verbose' in step[1].__dict__:
                    step[1].verbose = False
        
        verbose_level = 2 if self.verbose else 0
        
        # Perform cross-validation
        cv_results = cross_validate(
            estimator=pipe,
            X=df,
            y=y,
            cv=cv,
            scoring=self.scoring_metrics,
            return_train_score=True,
            verbose=verbose_level,
        )
        cv_results["type"] = type

        return cv_results

    def get_metrics(
        self,
        *,
        X,
        y_true,
        type: str,  # = 'holdout'
        prefix: str,  # = 'train' OR 'test'
        verbose=True,
        fmt=".4f",
    ):

        # y_pred = self.predict(X)
        # y_pred_proba = self.predict_proba(X)

        metrics = {
            "type": type,
        }
        for key in self.scoring_metrics:
            df = X.copy()
            metric_key = "_".join((prefix, key))
            metrics[metric_key] = self.scoring_metrics[key](
                estimator=self.pipe,
                X=df,
                y_true=y_true,
            )
            if verbose:
                print(f"{type} {metric_key}: {metrics[metric_key]:{fmt}}")

        return metrics

    def fit(self, X, y):
        X = X.copy()
        self.pipe.fit(X, y)

    def predict(self, X):
        X = X.copy()
        y_pred = self.pipe.predict(X)
        return y_pred

    def predict_proba(self, X):
        X = X.copy()
        y_pred_proba = self.pipe.predict_proba(X)
        return y_pred_proba

    def get_X_y(self, dataset):
        target = self._params["target"]
        X = dataset.drop(target, axis=1)
        # y = dataset[target].reset_index(drop=True)
        y = dataset[target].values

        return X, y

    def get_summary(self):
        estimator = self.pipe.steps[-1][-1]
        estimator_class_name = estimator.__class__.__name__
        if estimator_class_name == "StatsModelsEstimator":
            results = estimator.results_.summary()
            print(results)
            return
        print(f'no summary in estimator class "{estimator_class_name}"')


def _create_pipeline(
    *,
    estimator, # TODO: and init estimator here with these params! Add random_state here with check has_random_state and apply init_with_random_state!
    # source_features,
    # features_to_drop,
    minmax_features,
    passthrough_features,
    log_features,
    add_init_transformer=True,
    add_df_transformer=True,
    add_const=False,
    estimator_params:dict=None,
    base_estimator=None,
    base_estimator_params:dict=None,
    dim_transformer:TransformerMixin=None,
    dim_transformer_params:dict=None,
    add_smote=True,
    is_smotenc=False,
    smote_params:dict=None,
    random_state=RANDOM_STATE,
    std_features=None,  # If none, this features would be generated automatically
    verbose=True,
    **kwargs,
):
    pipeline = []

    if add_init_transformer:
        init_trans = InitialTransformer(
            # features_to_drop=features_to_drop,
            # add_sedimentation_sign=add_sedimentation_sign,
            log_features=log_features,
        )
        pipeline.append(("init_transformer", init_trans))

    ct = _get_column_transformer(
        # source_features=source_features,
        # features_to_drop=features_to_drop,
        minmax_features=minmax_features,
        passthrough_features=passthrough_features,
        std_features=std_features,
        verbose=verbose,
    )
    pipeline.append(("column_transformer", ct))

    if add_df_transformer:
        feature_names = _get_feature_names(ct)
        df_transformer = DataFrameTransformer(
            feature_names=feature_names,
            add_const=add_const,
        )
        pipeline.append(("df_transformer", df_transformer))

    if add_smote:
        smote_params = smote_params or {} # If None, use default SMOTE[NC] parameters
        
        if is_smotenc:
            smote_name = 'smotenc'
            smote_method = SMOTENC
        else:
            smote_name = 'smote'
            smote_method = SMOTE
        
        pipeline.append(
            (
                smote_name, 
                smote_method(
                    # If random_state is in smote_params, it will overwrite the random_state
                    **{
                        'random_state': random_state,
                        **smote_params,   
                    }
                )
            )
        )
    
    if dim_transformer is not None:
        dim_transformer_params = dim_transformer_params or {}
        pipeline.append(
            (
                dim_transformer.__name__,
                init_with_random_state(
                    dim_transformer,
                    random_state,
                    **dim_transformer_params,
                )
            )
        )
    
    estimator_params = estimator_params or {}

    # Process base_estimator if it is used
    if base_estimator is not None:
        base_estimator_params = base_estimator_params or {}
        base_estimator = init_with_random_state(
            base_estimator, random_state, **base_estimator_params
        )
        
        estimator_params = {
            **estimator_params,
            'base_estimator': base_estimator,
        }
        
    pipeline.append(
        (
            'estimator', 
            init_with_random_state(estimator, random_state, **estimator_params)
        )
    )
    
    return Pipeline(pipeline)


def has_random_state(
    cls,
    random_state_name:str='random_state',
):
    """Check if the class (of the estimator or transformer) has a random_state parameter in its constructor."""
    try:
        sig = inspect.signature(cls.__init__)
        return random_state_name in sig.parameters
    except (TypeError, ValueError):
        print(f'[WARNING] No signature was found in {cls} constructor (__init__)')
        return False

def init_with_random_state(cls, random_state, **estimator_params):
    """Initialize the estimator (or transformer) with a random state."""
    random_state_names = ['random_state', 'seed']
    for random_state_name in random_state_names:
        if has_random_state(cls, random_state_name=random_state_name):
            estimator_params = {
                random_state_name: random_state,
                **estimator_params,
            } # If random_state or SEED is in estimator_params, it will overwrite the random_state
    return cls(**estimator_params)


class AugmentedDataset(Dataset):
    def __init__(
        self,
        X_tensor:torch.Tensor,
        y_tensor:torch.Tensor,
        mixup_prob: float = 0.5,
        beta_distribution_params: tuple = (0.4, 0.4),
        noise_std: float = 0.01,
        apply_noise: bool = True,
    ):
        """Initialize an augmented dataset for mixup and optional noise augmentation.

        Args:
            X_tensor: Input features of shape (n_samples, n_features).
            y_tensor: Target labels of shape (n_samples,).
            mixup_prob: Probability of applying mixup augmentation. Defaults to 0.5.
            beta_distribution_params: Parameters (alpha, beta) for the Beta distribution used in mixup. Defaults to (0.4, 0.4).
            noise_std: Standard deviation of Gaussian noise to add. Defaults to 0.01.
            apply_noise: Whether to apply Gaussian noise augmentation. Defaults to True.

        Extra attributes:
            class_indices (dict): Mapping from each class label to indices for same-class mixup sampling.
        """
        self.X = X_tensor
        self.y = y_tensor
        
        self.mixup_prob = mixup_prob
        self.beta_distribution_params = beta_distribution_params
        self.noise_std = noise_std
        self.apply_noise = apply_noise
        
        self.class_indices = self._build_class_indices()
        
    def _build_class_indices(self):
        """Build indices for each class. It is used for Mixup with the same class samples."""
        sorted_y, sorted_idx = torch.sort(self.y)
        # Get unique classes and their counts
        unique_classes, counts = torch.unique_consecutive(
            sorted_y,
            return_counts=True,
        )
        
        class_indices = {}
        idx_start = 0
        # Get indices for each class based on sorted_idx and counts
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            idx_end = idx_start + count
            class_indices[cls] = sorted_idx[idx_start:idx_end]
            idx_start = idx_end
        
        return class_indices
      
        
    def __len__(self):
        return len(self.X)
    
    
    # NOTE: it mixes up only with the same class samples
    def __getitem__(self, idx:int):
        """Retrieve an augmented data sample with same-class mixup and optional Gaussian noise.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            torch.Tensor: Augmented feature tensor at the specified index.
        """
        
        x = self.X[idx]
        y = self.y[idx]
        
        if torch.rand(1).item() < self.mixup_prob:
            # Random second index for Mixup (from the same class)
            same_class_indices = self.class_indices[int(y.item())]
            # No mixup if there is only one sample in the class
            if len(same_class_indices) > 1:
                # Avoid mixing with self
                j = idx
                while j == idx:
                    j = same_class_indices[
                        int(
                            torch.randint(
                                0, len(same_class_indices), (1,)
                            ).item()
                        )
                    ]
                x2 = self.X[j]
                
                lam = np.random.beta(*self.beta_distribution_params)
                x = lam * x + (1 - lam) * x2
        
        if self.apply_noise:
            x += torch.randn_like(x) * self.noise_std
        
        return x, y
        

 
# TODO: Proceed checking and testing this model and BetaVAEncoder
# Variational Autoencoder Model
# Inspired by https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
class VAE(nn.Module):
    def __init__(
        self,
        input_dim:int,
        latent_dim:int=3, # could be changed to 4 or 5, if needed
        hidden_dim:int=32, # could be optimized
        normalization:str="batch", # could be "batch" or "layer", or False
        activation:str="LeakyReLU", # could be "ReLU" or "LeakyReLU", or False
        negative_slope:float=0.1, # only used if activation == "LeakyReLU"
        verbose:bool=True,
    ):
        super().__init__()
        
        encoder_layers = [
            nn.Linear(input_dim, hidden_dim),
        ]
        decoder_layers = [
            nn.Linear(latent_dim, hidden_dim),
        ]
        
        if normalization:
            if normalization == "batch":
                Norm = nn.BatchNorm1d
            elif normalization == "layer":
                Norm = nn.LayerNorm
            else:
                raise ValueError(
                    f"Normalization method {normalization} not supported"
                )
            # If no activation, then normalization is redundant before latent space,
            # since latent space makes normalization
            if activation:
                encoder_layers.append(Norm(hidden_dim))
            decoder_layers.append(Norm(hidden_dim))
        
        if activation:
            if activation == "ReLU":
                Activation = nn.ReLU
                activation_params = {
                    "inplace": True,
                }
            elif activation == "LeakyReLU":
                Activation = nn.LeakyReLU
                activation_params = {
                    "negative_slope": negative_slope,
                    "inplace": True,
                }
            else:
                raise ValueError(
                    f"Activation function {activation} not supported"
                )
            for layers in [encoder_layers, decoder_layers]:
                layers.append(Activation(**activation_params))
        
        decoder_layers.append(
            nn.Linear(hidden_dim, input_dim)
        )
        # Encoder: input -> hidden_dim
        # Transform input to latent space
        self.encoder = nn.Sequential(
            *encoder_layers,
        )
        
        # Latent space: hidden_dim -> 2 layers with latent_dim
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_sigma2 = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: latent_dim -> hidden_dim -> input
        # Reconstruct input from latent space
        self.decoder = nn.Sequential(
            *decoder_layers,
        )
        
        if verbose:
            print(f"Encoder: {self.encoder}")
            print(f"Latent space:")
            print(f"\tMu: {self.fc_mu}")
            print(f"\tlog_sigma2: {self.fc_log_sigma2}")
            print(f"Decoder: {self.decoder}")
    
    def encode(self, x):
        """Transform input to latent space"""
        enc_out = self.encoder(x)
        mu = self.fc_mu(enc_out)
        log_sigma2 = self.fc_log_sigma2(enc_out)
        return mu, log_sigma2
    
    def reparameterize(self, mu, log_sigma2):
        """Reparameterization trick, since direct sampling from latent space z~N(\mu, \sigma^2) is not differentiable"""
        std = torch.exp(0.5 * log_sigma2) # Compute standard deviation from log variance
        eps = torch.randn_like(std) # Sample some noise from standard normal distribution
        z = mu + eps * std # Return reparametrized latent vector z 
        return z # (z = mu + eps * std) allows to get gradients: dz/d\mu = 1, dz/d\sigma = eps
    
    def decode(self, z):
        """Reconstruct input from latent space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, log_sigma2 = self.encode(x) # Transform input to latent space
        z = self.reparameterize(mu, log_sigma2) # Reparameterize to get differentiable latent vector z
        x_reconstr = self.decode(z) # Decode latent vector z to reconstructed input x\hat
        return x_reconstr, mu, log_sigma2
    

class BetaVAEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        VAE_class:nn.Module=VAE,
        VAE_params:dict=None, # includes latent_dim, hidden_dim, normalization, activation, negative_slope. Input dim is set automatically
        batch_size:int=32,
        shuffle:bool=True,
        beta_start:float=0.0,
        beta_end:float=4.0,
        beta_warmup_steps:int=100,
        learning_rate:float=1e-3,
        scheduler_class:torch.optim.lr_scheduler.LRScheduler=torch.optim.lr_scheduler.ReduceLROnPlateau, # any scheduler from torch.optim.lr_scheduler or False
        scheduler_params:dict=None,
        max_epochs:int=100,
        early_stopping:bool=True,
        early_stopping_patience:int=10,
        early_stopping_min_delta:float=1e-3,
        augmented_dataset_params:dict=None,
        device_name:str=None,
        seed:int=RANDOM_STATE,
        verbose:bool=False,
    ):
        self.VAE_class = VAE_class
        # self.VAE_params = deepcopy(VAE_params) if VAE_params else {}
        self.VAE_params = {} if VAE_params is None else VAE_params
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_warmup_steps = beta_warmup_steps
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        # Prepare scheduler and its params
        self.scheduler_class = scheduler_class
        if (
            scheduler_class == torch.optim.lr_scheduler.ReduceLROnPlateau 
            and scheduler_params is None
        ):
            self.scheduler_params = {
                'mode': 'min',
                'patience': 5,
                'factor': 0.1,
                'min_lr': 1e-5,
            }
        else:
            self.scheduler_params = scheduler_params or {}
        
        self.device_name = device_name
        # Prepare device
        if self.device_name is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)
        self.verbose = verbose
        self.seed = seed
        if self.seed:
            self.set_seed(self.seed)
            
        # self.augmented_dataset_params = {
        #     "mixup_prob": 0.5,
        #     "beta_distribution_params": (0.4, 0.4),
        #     "noise_std": 0.01,
        #     "apply_noise": True,
        # }
        # self.VAE_params = {} if VAE_params is None else VAE_params
        self.augmented_dataset_params = {} if augmented_dataset_params is None else augmented_dataset_params
        
    def set_seed(self, seed:int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False # Auto-tuner of cuDNN would be disabled
            
    def _to_tensor(self, X, to_device:bool=False):
        """Convert pandas DataFrame or numpy array to Tensor and move to device if specified
        Args:
            X: source data
            to_device: whether to move the tensor to the device

        Returns:
            Tensor
        """
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if torch.is_tensor(X):
            X = X.to(torch.float32)
        else:
            X = torch.tensor(X, dtype=torch.float32)
        
        if to_device:
            X = X.to(self.device)
            
        return X
        
        
    def fit(self, X, y):
        # NOTE: y is used for Mixup only!
        if y is None:
            raise ValueError("y is None")
        
        input_dim = X.shape[1]

        # If scheduler Plateau or early stopping is used, split data into train and validation sets
        if (
            self.scheduler_class == torch.optim.lr_scheduler.ReduceLROnPlateau
            or self.early_stopping
        ):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, random_state=self.seed,
            )
            val_tensor = self._to_tensor(X_val, to_device=True)
        else:
            X_train = X
            val_tensor = None
        
        X_tensor = self._to_tensor(X_train, to_device=False) # Better to keep on CPU for the DataLoader
        # dataset = TensorDataset(X_tensor)
        y_tensor = self._to_tensor(y_train, to_device=False)
        
        dataset = AugmentedDataset(
            X_tensor=X_tensor,
            y_tensor=y_tensor,
            **self.augmented_dataset_params,
            # mixup_prob=self.mixup_prob,
            # beta_distribution_params=self.beta_distribution_params,
            # noise_std=self.noise_std,
            # apply_noise=self.apply_noise,
        )
        
        # DataLoader for training batches (with fixed seed for reproducibility)
        generator = torch.Generator()
        if self.seed:
            generator.manual_seed(self.seed)
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            generator=generator,
        )
        
        # VAE_params should be updated
        vae_params = {
            **self.VAE_params,
            'input_dim': input_dim,
            'verbose': self.verbose,
        }
        if self.verbose:
            print(f"VAE_params: {vae_params}")
        self.model = self.VAE_class(**vae_params).to(self.device)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        
        if self.scheduler_class:
            scheduler = self.scheduler_class(
                optimizer,
                **self.scheduler_params,
            )
        else:
            scheduler = None
        
        step = 0  # Initialize global step counter for beta warmup
        patience_counter = 0
        best_loss = float('inf')
        
        mean_loss = MeanMetric().to(self.device)
        mean_reconstr_loss = MeanMetric().to(self.device)
        mean_kl_divergence = MeanMetric().to(self.device)
        
        epoch_bar = trange(
            self.max_epochs,
            desc="Epochs",
            disable=not self.verbose,
        )
        
        all_metrics = []
        
        for epoch in epoch_bar:
            self.model.train() # Set model to training mode
            
            for batch in loader:
                X_batch, _ = batch
                X = X_batch.to(self.device)
                step += 1
                beta = min(
                    self.beta_end,
                    (
                        self.beta_start + (self.beta_end - self.beta_start) 
                        * step / self.beta_warmup_steps
                    ),
                )
                x_reconstr, mu, log_sigma2 = self.model(X)
                loss, reconstr_loss, kl_divergence = (
                    self.beta_vae_loss(x_reconstr, X, mu, log_sigma2, beta)
                )
                optimizer.zero_grad() # Reset gradients
                loss.backward() # Backpropagate loss
                optimizer.step() # Update model parameters
                
                # Current batch metrics
                mean_loss.update(loss.detach())
                # Other metrics already detached
                mean_reconstr_loss.update(reconstr_loss)
                mean_kl_divergence.update(kl_divergence)
                
            # Compute epoch metrics
            epoch_loss = mean_loss.compute().item()
            epoch_reconstr_loss = mean_reconstr_loss.compute().item()
            epoch_kl_divergence = mean_kl_divergence.compute().item()
            
            epoch_metrics = {
                'Epoch': epoch,
                'Total Loss': epoch_loss,
                'Reconstr. Loss': epoch_reconstr_loss,
                'KL divergence': epoch_kl_divergence,
                'beta': beta,
            }
            
            postfix = {
                'Total Loss': f"{epoch_loss:.4f}",
                'Reconstr. Loss': f"{epoch_reconstr_loss:.4f}",
                'KL divergence': f"{epoch_kl_divergence:.4f}",
                'beta': f"{beta:.2f}",
            }
            
            self.model.eval() # Set model to evaluation mode
            if val_tensor is not None:
                with torch.no_grad():
                    x_reconstr, mu, log_sigma2 = self.model(val_tensor)
                    val_loss, val_reconstr_loss, val_kl_divergence = (
                        self.beta_vae_loss(
                            x_reconstr, val_tensor, mu, log_sigma2, beta
                        )
                    )
                
                epoch_metrics.update({
                    'Val Total Loss': val_loss.item(),
                    'Val Reconstr. Loss': val_reconstr_loss.item(),
                    'Val KL divergence': val_kl_divergence.item(),
                }) 
                postfix.update({
                    'Val Total Loss': f"{val_loss.item():.4f}",
                    'Val Reconstr. Loss': f"{val_reconstr_loss.item():.4f}",
                    'Val KL divergence': f"{val_kl_divergence.item():.4f}",
                }) 
                    
                scheduler.step(val_loss.item())
            
            all_metrics.append(epoch_metrics)
            
            postfix_str = "\t".join([
                f"{key}: {value}" for key, value in postfix.items()
            ])
            
            epoch_bar.set_postfix_str(postfix_str)
            
            # Reset metrics
            mean_loss.reset()
            mean_reconstr_loss.reset()
            mean_kl_divergence.reset()
            
            # if self.early_stopping and beta == self.beta_end:
            if self.early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}.")
                    break
        
        if self.verbose:
            metrics_df = pd.DataFrame(all_metrics)
            fig, axes = plt.subplots(3, 1, figsize=(4, 7), sharex=True, dpi=300)
            
            total_loss_metric_names = ['Total Loss']
            reconstr_loss_metric_names = ['Reconstr. Loss']
            kl_divergence_metric_names = ['KL divergence']
            if val_tensor is not None:
                total_loss_metric_names.append('Val Total Loss')
                reconstr_loss_metric_names.append('Val Reconstr. Loss')
                kl_divergence_metric_names.append('Val KL divergence')
            
            metrics_df.plot(x='Epoch', y=total_loss_metric_names, ax=axes[0])
            
            metrics_df.plot(x='Epoch', y=reconstr_loss_metric_names, ax=axes[1])
            metrics_df.plot(x='Epoch', y=kl_divergence_metric_names, ax=axes[2])
            
            beta_ax = axes[2].twinx()
            metrics_df.plot(x='Epoch', y='beta', linestyle='--', color='black', ax=beta_ax)
            beta_ax.set_ylabel('Beta')
            beta_ax.legend(loc='best')
            
            axes[0].set_title('Total Loss')
            axes[1].set_title('Reconstr. Loss')
            axes[2].set_title('KL divergence')
            for i, ax in enumerate(axes):
                ax.legend(["Train", "Val"])
                ax.set_ylabel('Loss')
                ax.set_ylim(0, None)   
                
            fig.tight_layout()
                
    def transform(self, X):
        X_tensor = self._to_tensor(X, to_device=True)
        self.model.eval()
        with torch.no_grad():
            Z, _ = self.model.encode(X_tensor) # Need only mu
        return Z.cpu().numpy()
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def inverse_transform(self, Z):
        Z_tensor = self._to_tensor(Z, to_device=True)
        self.model.eval()
        with torch.no_grad():   
            X_reconstr = self.model.decode(Z_tensor) # Z is latent space vector (batch of vectors) z
        return X_reconstr.cpu().numpy()
    
    def beta_vae_loss(self, x_reconstr, x, mu, log_sigma2, beta=4.0):
        # beta can be from 1 to 10
        reconstr_loss = F.mse_loss(x_reconstr, x, reduction='mean')
        # Kullback-Leibler divergence
        kl_divergence = -0.5 * torch.mean(1 + log_sigma2 - mu.pow(2) - log_sigma2.exp()) # Compute KL divergence to N(0,1)
        # Total loss with beta weight on KL divergence
        loss = reconstr_loss + beta * kl_divergence
        
        return loss, reconstr_loss.detach(), kl_divergence.detach()
       

# Wrapper for pytorch_tabular
class PytorchTabularEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        model_class,
        model_config_params:dict=None, # layers, activation, etc.
        data_config_params:dict=None, # continuous_cols, categorical_cols, etc.
        trainer_config_params:dict=None, # gpus, max_epochs, etc.
        optimizer_config_params:dict=None, # optimizer, lr_scheduler, etc.
        seed:int=RANDOM_STATE,
        suppress_lightning_logger:bool=True,
        verbose:bool=False,
    ):
        self.model_class = model_class
        self.__name__ = model_class.__name__
        self.model_class = model_class
        self.model_config_params = model_config_params
        self.data_config_params = data_config_params
        self.trainer_config_params = trainer_config_params
        self.optimizer_config_params = optimizer_config_params
        self.seed = seed
        self.suppress_lightning_logger = suppress_lightning_logger
        self.verbose = verbose
        
        # self.logger = TensorBoardLogger(save_dir="../logs/", name="drop-impact-exp")
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        df = pd.DataFrame(X)
        df['target'] = y
        
        data_config_params = {
            'target': ['target'],
            'continuous_cols': list(df.columns[:-1]),
            'categorical_cols': [],
            **(self.data_config_params or {}), # continuous_cols and categorical_cols will be overwritten if specified in data_config_params
        }
        data_config = DataConfig(**data_config_params)
        
        model_config_params = {
            'task': 'classification',
            'metrics': METRICS,
            **(self.model_config_params or {}), # task and metrics will be overwritten if specified in model_config_params
        }
        if 'activation' in model_config_params:
            if model_config_params['activation'] == 'ELU':
                model_config_params['initialization'] = 'xavier'
            else:
                model_config_params['initialization'] = 'kaiming'
        model_config = self.model_class(**model_config_params)
        
        trainer_config_params = {
            'seed': self.seed,
            **(self.trainer_config_params or {}), # seed be overwritten if specified in trainer_config_params
        }
        trainer_config = TrainerConfig(**trainer_config_params)
        
        optimizer_config = OptimizerConfig(
            **(self.optimizer_config_params or {}),
        )
        
        self.model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            verbose=self.verbose,
            suppress_lightning_logger=self.suppress_lightning_logger,
        )

        self.model.fit(
            train=df,
        )
        # self.model.trainer.logger = self.logger
        
        return self
    
    def predict(self, X):
        df = pd.DataFrame(X)
        preds = self.model.predict(df)
        y_pred = preds.values[:,-1]
        return y_pred
    
    def predict_proba(self, X):
        df = pd.DataFrame(X)
        preds = self.model.predict(df)
        y_pred_proba = preds.values[:,:-1] # Remove last column (prediction)
        return y_pred_proba
    
        

# Wrapper for Statsmodel
class StatsModelsEstimator(BaseEstimator):
    def __init__(self, model_class, verbose=True, **init_params):
        self.model_class = model_class
        self.__name__ = model_class.__name__
        self.init_params = init_params
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y)
        self.model_ = self.model_class(endog=y, exog=X, **self.init_params)
        # Get fit_method. Pass "fit", if fit_method did not specified
        fit_method = fit_params.pop("fit_method", "fit")
        # getattr - retrieve proper method with name `fit_method`,
        # Then, apply this method with **fit_params
        fit_fn = getattr(self.model_, fit_method)
        
        if self.verbose:
            self.results_ = fit_fn(**fit_params)
        else:
            # Redirect stdout to None to suppress output
            with contextlib.redirect_stdout(io.StringIO()):
                self.results_ = fit_fn(**fit_params)
        return self

    def predict(self, X, level=0.5, **predict_params):
        # Get probabilities only for main class "1"
        y_pred_proba = self.predict_proba(X, **predict_params)[:, 1]

        y_pred = np.zeros_like(y_pred_proba)
        y_pred[y_pred_proba > level] = 1

        return y_pred

    def predict_proba(self, X, **predict_params):
        
        probs = self.results_.predict(exog=X, **predict_params)
        if hasattr(probs, 'to_numpy'):
            probs = probs.to_numpy()

        prob = (
            probs.reshape((-1, 1))
        )
        y_pred_proba = np.hstack([1 - prob, prob])
        # y_pred_proba = prob
        return y_pred_proba


class DecisionStumpEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, less_sign=True, **init_params):
        self.less_sign = less_sign
        self.__name__ = 'DecisionStumpEstimator'

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, **predict_params):
        kl = (649 + 3.76 / (X["relative_roughness"] ** 0.63)) ** (5 / 8)
        if self.less_sign:
            y_pred = np.where(X["K"] < kl, 1, 0)
        else:
            y_pred = np.where(X["K"] >= kl, 1, 0)
        return y_pred

    def predict_proba(self, X):
        pred = self.predict(X)
        return np.vstack([1 - pred, pred]).T
        # pred = self.predict(X)
        # proba = np.zeros((len(pred), len(self.classes_)))
        # for i, cls in enumerate(self.classes_):
        #     proba[:, i] = (pred == cls).astype(int)
        # return proba


# Custom transformer to convert NumPy array to DataFrame with feature names
class InitialTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        # features_to_drop,
        # add_sedimentation_sign=False,
        log_features,
    ):
        self.log_features = log_features

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns
        return self  # Nothing to fit here

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input to InitialTransformer must be a pandas DataFrame")

        X = X.copy()
        # Get logarithm of features
        for log_feature in self.log_features:
            X[log_feature] = np.log10(X[log_feature] + 1e-15)

        # if self.log_roughness:
        #     X['relative_roughness'] = np.log10(X['relative_roughness'])

        # # Get logarithm of sedimentation Stokes number
        # if self.log_sedimentation_Stk:
        #     X['sedimentation_Stk'] = np.log10(X['sedimentation_Stk'] + 1e-15)

        # # Drop features
        # if self.features_to_drop:
        #     # Check if columns to drop exist in the DataFrame
        #     for col in self.features_to_drop:
        #         if col not in X.columns:
        #             raise ValueError(f"Column '{col}' is not present in the DataFrame.")
        #     X = X.drop(self.features_to_drop, axis=1)

        return X


# def _add_sedimentation_sign(row, column='sedimentation_Re'):
#     if row['particle_liquid_density_ratio'] < 1.:
#         row[column] *= -1
#     return row

# def _prepare_df(self, df, features_to_drop):
#     df = df.copy()
#     # Add sign to the sedimentation_Re
#     df = df.apply(
#         self._add_sedimentation_sign,
#         axis=1,
#     )
#     # Get logarithm of relative roughness
#     df['relative_roughness'] = np.log10(df['relative_roughness'])
#     # Drop features
#     df = df.drop(features_to_drop, axis=1)

#     return df


# Custom transformer to convert NumPy array to DataFrame with feature names
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names, add_const=True):
        self.feature_names = feature_names
        self.add_const = add_const

    def fit(self, X, y=None):
        return self  # Nothing to fit here

    def transform(self, X):
        X = X.copy()
        if self.add_const:
            X = sm.add_constant(X)
            columns = ["const"] + self.feature_names
        else:
            columns = self.feature_names
        # Convert the NumPy array back into a DataFrame with the feature names
        return pd.DataFrame(X, columns=columns)


# Function to extract feature names after ColumnTransformer
def _get_feature_names(column_transformer):
    # List to store the final feature names
    feature_names = []

    # Loop through each transformer in the ColumnTransformer
    for name, transformer, columns in column_transformer.transformers:
        if transformer != "drop":
            if hasattr(transformer, "get_feature_names_out"):
                # If transformer supports get_feature_names_out (e.g., OneHotEncoder)
                feature_names.extend(transformer.get_feature_names_out(columns))
            else:
                # Otherwise, just use the original column names (e.g., for StandardScaler)
                feature_names.extend(columns)

    return feature_names


def _get_column_transformer(
    *,
    # source_features,
    # features_to_drop,
    minmax_features,
    passthrough_features,
    std_features=None,
    verbose=True,
):

    transformers = [
        ("minmax", MinMaxScaler(), minmax_features),
        # (
        #     'minmax_neg',
        #     MinMaxScaler(feature_range=(-1,1)),
        #     minmax_neg_features
        # ),
        ("std", StandardScaler(), std_features),
        ("passthrough", "passthrough", passthrough_features),
    ]

    # NOTE: Features from `features_to_drop` will be dropped automatically!
    # Since by default remainder = 'drop'
    # if features_to_drop:
    #     transformers.insert(
    #         0,
    #         ('feature_dropper', 'drop', features_to_drop),
    #     )

    # TODO: Add one-feature fit Scaler, which fits on volume fraction
    # and applied to init_volume_fraction and volume_fraction
    ct = ColumnTransformer(
        transformers=transformers,
    )

    if verbose:
        display(ct)

    return ct


def _drop_features(features, features_to_drop, inplace=False):
    if not (inplace):
        features = list(features)
    for drop in features_to_drop:
        if drop in features:
            features.remove(drop)
    if inplace:
        return
    return tuple(features)


def deep_update_dim_transformer_params(
    ml_pipe:MLPipeline,
    suggested_params:dict,
)->dict:
    """Deep update of the dimension transformer parameters.
    """
    original_params = deepcopy(
        ml_pipe._pipeline_params['init_dim_transformer_params']
    )
    deep_update(original_params, suggested_params)
    
    return original_params


def deep_update_estimator_params(
    ml_pipe:MLPipeline,
    suggested_params:dict,
)->dict:
    """Deep update of the estimator parameters.
    
    Args:
        ml_pipe: An instance of MLPipeline used for model training and evaluation.
        suggested_params: A dictionary containing the suggested hyperparameters.
    
    Returns:
        A dictionary containing the estimator parameters (including unchanged).
    """
    original_params = deepcopy(
        ml_pipe._pipeline_params['init_estimator_params']
    )
    deep_update(original_params, suggested_params)
    
    return original_params


def deep_update(original, updates):
    """Deep update of the original dictionary.
    
    Args:
        original: Original dictionary.
        updates: Dictionary with updates.
    """
    
    for key, value in updates.items():
        if isinstance(value, dict) and key in original:
            deep_update(original[key], value)
        else:
            original[key] = value

def update_estimator_params(
    ml_pipe:MLPipeline,
    suggested_params:dict,
    estimator_type:str='main',
) -> dict:
    """Upate the estimator parameters based the parameters from pipeline.

    Args:
        ml_pipe: An instance of MLPipeline used for model training and evaluation.
        suggested_params: A dictionary containing the suggested hyperparameters.
    
    Returns:
        A dictionary containing the estimator parameters.
    """
    if estimator_type == 'main':
        estimator_params = ml_pipe._pipeline_params['init_estimator_params'].copy()
        estimator_params.update(suggested_params)
    elif estimator_type == 'base':
        estimator_params = ml_pipe._pipeline_params['init_base_estimator_params'].copy()
        estimator_params.update(suggested_params)
    else:
        raise ValueError(f"Invalid estimator type: {estimator_type}")
    return estimator_params


def update_smote_params(
    ml_pipe:MLPipeline,
    suggested_params:dict,
) -> dict:
    """Upate the SMOTE parameters based the parameters from pipeline.

    Args:
        ml_pipe: An instance of MLPipeline used for model training and evaluation.
        suggested_params: A dictionary containing the suggested SMOTE-hyperparameters.
    
    Returns:
        A dictionary containing the SMOTE parameters.
    """
    smote_params = ml_pipe._pipeline_params['smote_params']
    smote_params.update(suggested_params)
    return smote_params

def smote_objective(trial:optuna.trial.Trial, ml_pipe:MLPipeline):
    """Objective function for SMOTE optimization using Optuna.
    NOTE: With this parameters for SMOTE no need to use Optuna. GridSearchOptimizer is enough.

    Args:
        trial: An Optuna trial object used to suggest hyperparameters.
        ml_pipe: An instance of MLPipeline used for training and evaluation.
    """
    
    suggested_smote_params = {
        'k_neighbors': trial.suggest_int('k_neighbors', 3, 10),
        # 'sampling_strategy': trial.suggest_categorical(
        #     'sampling_strategy', [0.6, 0.7, 0.8, 0.9, 1.0]
        # ),
        'sampling_strategy': trial.suggest_float(
            'sampling_strategy', 0.60, 1.00, step=0.01),
    }
    
    smote_params = update_smote_params(ml_pipe, suggested_smote_params)
    
    # Conduct the step with the suggested SMOTE parameters and default estimator
    score = ml_pipe.step(
        estimator=ml_pipe._pipeline_params['estimator'],
        smote_params=smote_params,
    )
    
    return score


def pure_smote_objective(suggested_smote_params, ml_pipe:MLPipeline):
    """Objective function for SMOTE optimization using Optuna.

    Args:
        suggested_smote_params: Parameters for SMOTE.
        ml_pipe: An instance of MLPipeline used for training and evaluation.
    """
    
    smote_params = update_smote_params(ml_pipe, suggested_smote_params)
    
    # Conduct the step with the suggested SMOTE parameters and default estimator
    score = ml_pipe.step(
        estimator=ml_pipe._pipeline_params['estimator'],
        smote_params=smote_params,
    )
    
    return score


if __name__ == "__main__":
    estimator = StatsModelsEstimator
    estimator_params = {
        'model_class': Logit,
    }

    ml_pipe = MLPipeline(
        target="splashing",
        estimator=estimator,
        estimator_params=estimator_params,
        features_to_drop=(
            "Re",
            "We",
            "init_volume_fraction",
            "particle_droplet_diameter_ratio",
            "sedimentation_Re",
            # 'particle_liquid_density_ratio',
            "sedimentation_Stk",
            # 'sign_sedimentation_Re',
            # 'volume_fraction',
            # 'relative_roughness',
            # 'inclination',
            # 'wettability',
        ),
    )

    ml_pipe.run(
        save_model_and_metrics=False,
    )
