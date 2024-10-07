import pandas as pd
import numpy as np

from pathlib import Path
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    make_scorer,
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score, 
    roc_auc_score
)
from sklearn.model_selection import cross_validate, StratifiedKFold

import statsmodels.api as sm
from statsmodels.api import Logit

from IPython.display import display

sys.path.append(
    '../'
)
# from utils_functionality.split_utils.split_tools import load_df, get_train_test
from utils_functionality.split_utils.split_tools import load_df, get_train_test

class MLPipeline:
    def __init__(
        self,
        *,
        target,
        estimator,
        features_to_drop=(
            'Re', 
            'We', 
            'init_volume_fraction',
            'particle_droplet_diameter_ratio', 
            # 'volume_fraction', 
            # 'sedimentation_Re', 
            # 'relative_roughness', 
            # 'inclination',
            # 'wettability',
            # 'particle_liquid_density_ratio',
        ),
        minmax_features=(
            'inclination',
            'volume_fraction',
        ),
        passthrough_features=(
            'wettability',
        ),
        std_features=None,
        dataset_filename='df_dimless',
        path_data=Path('..', 'data'),
        targets=('splashing', 'no_fragmentation'),
        add_init_transformer=True,
        log_roughness=True,
        add_df_transformer=True,
        add_const=False,
        verbose=True,
    ):
        target_set = set(targets)
        self._params = {
            'target': target,
            'dataset_filename': dataset_filename,
            'path_data': path_data,
            'target_set': target_set,
        }
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
        
        self._pipeline_params = {
            'estimator': estimator,
            'source_features': source_features,
            'minmax_features': minmax_features,
            'passthrough_features': passthrough_features,
            'std_features': std_features,
            'features_to_drop': features_to_drop,
            'add_init_transformer': add_init_transformer,
            'log_roughness': log_roughness,
            'add_df_transformer': add_df_transformer,
            'add_const': add_const,
        }
        
        # Create full pipeline
        self.pipe = _create_pipeline(
            **self._pipeline_params
        )
        
        # NOTE: in new sklearn versions use response_method parameter instead of needs_proba
        self.scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
        }
        
        self.metric_results = []
    
    
    def run(self, verbose=True, random_state=42):
        
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
                cv_folds=5,
                random_state=random_state,
                type='cv',
            )
        )
        
        # Fit on holdout train dataset and get summary (if applicable)
        self.fit(
            X=X_train,
            y=y_train,
        )
        self.get_summary()
        
        # Predict on train, test and save metrics
        metric_results_list.append(
            self.get_metrics(
                X=X_train,
                y_true=y_train,
                type='holdout',
                verbose=False,
                prefix='train',
            )
        )
        metric_results_list.append(
            self.get_metrics(
                X=X_test,
                y_true=y_test,
                type='holdout',
                verbose=False,
                prefix='test',
            )
        )
        
        # Transform metric_results_list to dict and append to metric_results
        metric_results_dict = {}
        for metrics in metric_results_list:
            type = metrics['type']
            
            for key in metrics:
                if key != 'type':
                    metric_results_dict['_'.join((type, key))] = metrics[key]
        self.metric_results.append(metric_results_dict)
        
        # Prepare dataframe of final metrics
        self.metric_results_df = pd.DataFrame(self.metric_results)
        display(self.metric_results_df.T)
        
        # TODO: Save metrics and model
        
    
    def get_cv_metrics(
        self,
        *,
        X,
        y,
        cv_folds=5,
        random_state=None,
        shuffle=True,
        type:str='cv',
        verbose=True,
        fmt='.4f',
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
        cv_results['type'] = type
        
        return cv_results
    
    
    def get_metrics(
        self,
        *,
        X,
        y_true,
        type:str, # = 'holdout'
        prefix:str, # = 'train' OR 'test'
        verbose=True,
        fmt='.4f',
    ):
        
        # y_pred = self.predict(X)
        # y_pred_proba = self.predict_proba(X)
        
        metrics = {
            'type': type,
        }
        for key in self.scoring_metrics:
            df = X.copy()
            metric_key = '_'.join((prefix, key))
            metrics[metric_key] = self.scoring_metrics[key](
                estimator=self.pipe,
                X=df,
                y_true=y_true,
            )
            if verbose:
                print(f'{type} {metric_key}: {metrics[metric_key]:{fmt}}')
    
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
        target = self._params['target']
        X = dataset.drop(target, axis=1)
        # y = dataset[target].reset_index(drop=True)
        y = dataset[target].values
        
        return X, y
        
    
    def get_summary(self):
        estimator = self.pipe.steps[-1][-1]
        estimator_class_name = estimator.__class__.__name__
        if estimator_class_name == 'StatsModelsEstimator':
            results = estimator.results_.summary()
            print(results)
            return
        print(f'no summary in estimator class "{estimator_class_name}"')


def _create_pipeline(
    *,
    estimator,
    source_features,
    minmax_features,
    passthrough_features,
    features_to_drop,
    add_init_transformer=True,
    # add_sedimentation_sign=False, # do not forget
    log_roughness=True,
    log_sedimentation_Stk=True,
    add_df_transformer=True,
    add_const=False,
    std_features=None, # If none, this features would be generated automatically
    verbose=True,
):
    pipeline = []
    
    if add_init_transformer:
        init_trans = InitialTransformer(
            # features_to_drop=features_to_drop,
            # add_sedimentation_sign=add_sedimentation_sign,
            log_roughness=log_roughness,
            log_sedimentation_Stk=log_sedimentation_Stk,
        )
        pipeline.append(
            ('init_transformer', init_trans)
        )
    
    ct = _get_column_transformer(
        source_features=source_features,
        minmax_features=minmax_features,
        passthrough_features=passthrough_features,
        features_to_drop=features_to_drop,
        std_features=std_features,
        verbose=verbose,
    )
    pipeline.append(
        ('column_transformer', ct)
    )
    
    if add_df_transformer:
        feature_names = _get_feature_names(ct)
        df_transformer = DataFrameTransformer(
            feature_names=feature_names, 
            add_const=add_const,
        )
        pipeline.append(
            ('df_transformer', df_transformer)
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
        self.model_ = self.model_class(endog = y, exog = X, **self.init_params)
        # Get fit_method. Pass "fit", if fit_method did not specified
        fit_method = fit_params.pop("fit_method", "fit")
        # getattr - retrieve proper method with name `fit_method`, 
        # Then, apply this method with **fit_params
        self.results_ = getattr(self.model_, fit_method)(**fit_params)
        return self

    def predict(self, X, level=0.5, **predict_params):
        # Get probabilities only for main class "1"
        y_pred_proba = self.predict_proba(X, **predict_params)[:,1]
        
        y_pred = np.zeros_like(y_pred_proba)
        y_pred[y_pred_proba>level] = 1
        
        return y_pred

    def predict_proba(self, X, **predict_params):
        prob = (
            self.results_
            .predict(exog=X, **predict_params)
            .to_numpy()
            .reshape((-1,1))
        )
        y_pred_proba = np.hstack([1 - prob, prob])
        # y_pred_proba = prob
        
        return y_pred_proba


# Custom transformer to convert NumPy array to DataFrame with feature names
class InitialTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        # features_to_drop,
        # add_sedimentation_sign=False, 
        log_roughness,
        log_sedimentation_Stk,
    ):
        # self.features_to_drop = features_to_drop
        # self.add_sedimentation_sign = add_sedimentation_sign
        self.log_roughness = log_roughness
        self.log_sedimentation_Stk = log_sedimentation_Stk
    
    def fit(self, X, y=None):
        return self  # Nothing to fit here
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input to InitialTransformer must be a pandas DataFrame")
        # # Add sign to the sedimentation_Re
        # if self.add_sedimentation_sign:
        #     X = X.apply(
        #         _add_sedimentation_sign,
        #         axis=1,
        #     )
        
        X = X.copy()
        # Get logarithm of relative roughness
        if self.log_roughness:
            X['relative_roughness'] = np.log10(X['relative_roughness'])
        
        # Get logarithm of sedimentation Stokes number
        if self.log_sedimentation_Stk:
            X['sedimentation_Stk'] = np.log10(X['sedimentation_Stk'] + 1e-15)
        
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
            columns = ['const']+self.feature_names
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
        if transformer != 'drop':
            if hasattr(transformer, 'get_feature_names_out'):
                # If transformer supports get_feature_names_out (e.g., OneHotEncoder)
                feature_names.extend(
                    transformer.get_feature_names_out(columns)
                )
            else:
                # Otherwise, just use the original column names (e.g., for StandardScaler)
                feature_names.extend(columns)
    
    return feature_names


def _get_column_transformer(
    *,
    source_features,
    minmax_features,
    passthrough_features,
    features_to_drop,
    std_features=None,
    verbose=True,
):
    minmax_features = _drop_features(minmax_features, features_to_drop)

    if std_features is None:
        features_to_drop_std = (
            features_to_drop
            + minmax_features
            + passthrough_features
            # + minmax_neg_features  
            # + [target]
        )
        std_features = _drop_features(source_features, features_to_drop_std)

        if verbose:
            print('std_features')
            display(std_features)

    transformers = [
        ('minmax', MinMaxScaler(), minmax_features),
        # (
        #     'minmax_neg', 
        #     MinMaxScaler(feature_range=(-1,1)), 
        #     minmax_neg_features
        # ),
        ('std', StandardScaler(), std_features),
        ('passthrough', 'passthrough', passthrough_features),
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
    if not(inplace):
        features = list(features)
    for drop in features_to_drop:
        if drop in features:
            features.remove(drop)
    if inplace:
        return
    return tuple(features)


if __name__ == '__main__':
    estimator = StatsModelsEstimator(Logit)

    ml_pipe = MLPipeline(
        target='splashing',
        estimator=estimator,
        features_to_drop = (
            'Re', 
            'We', 
            'init_volume_fraction',
            'particle_droplet_diameter_ratio', 
            'sedimentation_Re',
            # 'particle_liquid_density_ratio',
            'sedimentation_Stk'
            # 'sign_sedimentation_Re',
            # 'volume_fraction', 
            # 'relative_roughness', 
            # 'inclination',
            # 'wettability',
        ),
    )

    ml_pipe.run()