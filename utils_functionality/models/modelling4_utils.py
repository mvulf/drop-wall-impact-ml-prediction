import pandas as pd
import numpy as np

from pathlib import Path
import sys
import os
import joblib

from collections.abc import Iterable

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
from sklearn.model_selection import cross_validate, StratifiedKFold

import statsmodels.api as sm
from statsmodels.api import Logit

from IPython.display import display

sys.path.append("../")
# from utils_functionality.split_utils.split_tools import load_df, get_train_test
from utils_functionality.split_utils.split_tools import load_df, get_train_test

RANDOM_STATE = 42


class MLPipeline:
    def __init__(
        self,
        *,
        target,
        estimator,
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
        path_results=Path("..", "results"),
        models_folder="models_modelling4",
        metrics_file="metrics_modelling4.xlsx",
    ):
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

        if estimator_class_name == "DecisionStumpEstimator":
            estimator_class_name = "DecisionStump"

        self.model_name = "_".join([estimator_class_name, self._params["target"]])
        if self.model_postfix:
            self.model_name = "_".join([self.model_name, self.model_postfix])

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

        self.metric_results = []

    def run(
        self,
        verbose=True,
        random_state=RANDOM_STATE,
        save_model_and_metrics=True,
    ):

        # Split X, y for fitting and predicting
        X_train, y_train = self.get_X_y(self.train)
        X_test, y_test = self.get_X_y(self.test)
        X, y = self.get_X_y(self.full_df)

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
        df["params"] = str(self._pipeline_params)

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
        type: str = "cv",
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
    estimator,
    # source_features,
    # features_to_drop,
    minmax_features,
    passthrough_features,
    log_features,
    add_init_transformer=True,
    add_df_transformer=True,
    add_const=False,
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
    
    pipeline.append(
        ('estimator', estimator)
    )
    
    return Pipeline(pipeline)


# Wrapper for Statsmodel
class StatsModelsEstimator(BaseEstimator):
    def __init__(self, model_class, **init_params):
        self.model_class = model_class
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


if __name__ == "__main__":
    estimator = StatsModelsEstimator(Logit)

    ml_pipe = MLPipeline(
        target="splashing",
        estimator=estimator,
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

    ml_pipe.run()
