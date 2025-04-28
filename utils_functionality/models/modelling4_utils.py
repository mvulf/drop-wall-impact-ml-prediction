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

from pathlib import Path
import sys
import os
import joblib

import inspect

from tqdm import tqdm

from collections.abc import Iterable
from functools import partial
import copy
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

# from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
from sklearn.model_selection import cross_validate, StratifiedKFold, ParameterGrid

import statsmodels.api as sm
from statsmodels.api import Logit

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig, TabNetModelConfig

import optuna

from IPython.display import display

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
        
    def optimize(self, n_trials:int):
        """Optimize the objective function.

        Args:
            n_trials: The number of trials to run for optimization.

        Returns:
            The study object containing the optimization results.
        """
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
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
            
            
        # Add features choice depending on the target
        if minmax_features is None:
            if target == "splashing":
                minmax_features = (
                    "inclination",
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
        )

        # Split train and test
        self.train, self.test = get_train_test(
            df=self.full_df,
            target=target,
            path_data=path_data,
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
        self.pipe = _create_pipeline(**self._pipeline_params)

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
        
    def step(
        self,
        estimator,
        estimator_params:dict=None,
        base_estimator=None,
        base_estimator_params:dict=None,
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
            
        # Create full pipeline
        self.pipe = _create_pipeline(**self._pipeline_params)
        
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
        random_state=RANDOM_STATE,
        save_model_and_metrics=True,
    ):

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
            )
        )

        # Fit on holdout train dataset and get summary (if applicable)
        self.fit(
            X=X_train,
            y=y_train,
        )
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
        params_dict = self._pipeline_params.copy()
        params_dict['estimator'] = params_dict['estimator'].__name__
        # If model_class is in params_dict['estimator_params'], 
        # replace model_class in estimator_params with its name
        if 'model_class' in params_dict['estimator_params']:
            params_dict['estimator_params']['model_class'] = (
                params_dict['estimator_params']['model_class'].__name__
            )
        # Remove init_estimator_params from params_dict
        params_dict.pop('init_estimator_params', None)
        params_dict.pop('init_base_estimator_params', None)
        # Process base_estimator if it is used
        if params_dict['base_estimator'] is not None:
            params_dict['base_estimator'] = params_dict['base_estimator'].__name__
        else:
            params_dict.pop('base_estimator', None)
            params_dict.pop('base_estimator_params', None)
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

        # Perform cross-validation
        cv_results = cross_validate(
            estimator=self.pipe,
            X=df,
            y=y,
            cv=cv,
            scoring=self.scoring_metrics,
            return_train_score=True,
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
    """Check if the class (of the estimator) has a random_state parameter in its constructor."""
    try:
        sig = inspect.signature(cls.__init__)
        return random_state_name in sig.parameters
    except (TypeError, ValueError):
        print(f'[WARNING] No signature was found in {cls} constructor (__init__)')
        return False

def init_with_random_state(cls, random_state, **estimator_params):
    """Initialize the estimator with a random state."""
    random_state_names = ['random_state', 'seed']
    for random_state_name in random_state_names:
        if has_random_state(cls, random_state_name=random_state_name):
            estimator_params = {
                random_state_name: random_state,
                **estimator_params,
            } # If random_state or SEED is in estimator_params, it will overwrite the random_state
    return cls(**estimator_params)


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
    def __init__(self, model_class, **init_params):
        self.model_class = model_class
        self.__name__ = model_class.__name__
        self.init_params = init_params

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y)
        self.model_ = self.model_class(endog=y, exog=X, **self.init_params)
        # Get fit_method. Pass "fit", if fit_method did not specified
        fit_method = fit_params.pop("fit_method", "fit")
        # getattr - retrieve proper method with name `fit_method`,
        # Then, apply this method with **fit_params
        self.results_ = getattr(self.model_, fit_method)(**fit_params)
        return self

    def predict(self, X, level=0.5, **predict_params):
        # Get probabilities only for main class "1"
        y_pred_proba = self.predict_proba(X, **predict_params)[:, 1]

        y_pred = np.zeros_like(y_pred_proba)
        y_pred[y_pred_proba > level] = 1

        return y_pred

    def predict_proba(self, X, **predict_params):
        prob = (
            self.results_.predict(exog=X, **predict_params).to_numpy().reshape((-1, 1))
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
    original_params = copy.deepcopy(
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
