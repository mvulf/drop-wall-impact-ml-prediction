import numpy as np

from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


dict_num_features = {
    'df_full': ['droplet_diameter', 'particle_droplet_diameter_ratio',
                      'particle_liquid_density_ratio', 'Re', 'We', 'We_Re',
                      'roughness', 'liquid_density',
                        'surface_tension', 'viscosity', 'particle_mean_diameter',
                        'particle_density', 'inclination', 'velocity'],
    'df_modelling_dimensionless_pf': ['Re^(1/4)', 'We^(1/4)^2', 'We^(1/4)^2 Re^(1/4)',
                         'inclination',
                        'particle_liquid_density_ratio',
                        'particle_droplet_diameter_ratio'],
    'df_modelling_dimensionless': ['Re', 'We', 'We_Re',
                         'inclination',
                        'particle_liquid_density_ratio',
                        'particle_droplet_diameter_ratio'],
    'df_modelling_no_multicollinearity_pf': ['droplet_diameter', 'particle_droplet_diameter_ratio',
                      'particle_liquid_density_ratio', 'Re^(1/4)', 'We^(1/4)^2', 
                      'We^(1/4)^2 Re^(1/4)',
                       'liquid_density',
                        'surface_tension', 'inclination'],
    'df_modelling_no_multicollinearity': ['droplet_diameter', 'particle_droplet_diameter_ratio',
                      'particle_liquid_density_ratio', 'Re', 'We', 'We_Re',
                       'liquid_density',
                        'surface_tension', 'viscosity', 'inclination']
}


def get_params(trial, model_str, random_state, cat_features=['wettability']):
    if 'catboostclassifier' in model_str:
        params = {
            'verbose':False,
            'random_seed': random_state,
            "objective": trial.suggest_categorical(
                "objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])}
        if params["bootstrap_type"] == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10)
        elif params["bootstrap_type"] == "Bernoulli":
            params["subsample"] = trial.suggest_float("subsample", 0.1, 1)
        if 'ohe' not in model_str: params['cat_features'] = cat_features
    if 'kneighborsclassifier' in model_str:
        params = {
            'random_state': random_state,
            "algorithm": trial.suggest_categorical(
                "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
            ),
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", 
                                        ["euclidean", "manhattan", "minkowski"])
        }
    if 'svc' in model_str:
        params = {
            'random_state': random_state,
            'C': trial.suggest_float('C', 1e-5, 1e5, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_float('gamma', 1e-5, 1e5, log=True),
            # 'degree': trial.suggest_int('degree', 2, 4),
            'coef0': trial.suggest_float('coef0', -1.0, 1.0)}
    if 'logisticregression' in model_str:
        params = {
            'random_state': random_state,
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'C': trial.suggest_float('C', 0.001, 10, log=True),
            'solver': trial.suggest_categorical('solver', ['saga', 'liblinear']),
            'max_iter': trial.suggest_int('max_iter', 100, 300),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }
    if 'xgbclassifier' in model_str: 
        params = {
            'random_state': random_state,
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)}
    return params


def get_model(model_str, params):
    dict_model = {
        'logistic': LogisticRegression,
        'svc': SVC,
        'kneighbors': KNeighborsClassifier,
        'xgbclassifier': XGBClassifier,
        'catboost': CatBoostClassifier}
    for key in dict_model.keys():
        if key in model_str: return dict_model[key](**params)


def get_preproc_pipeline(model_str, num_features, random_state, cat_features=['wettability']):
    pipeline, transformers = [], []
    if 'boost' not in model_str:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
        transformers = [("num", numeric_transformer, num_features)]
    if ('onehot' in model_str) and ('boost' not in model_str):
        categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(('cat', categorical_transformer, cat_features))
    if 'ordenc' in model_str:
        categorical_transformer = Pipeline(
            steps=[('ordenc', OrdinalEncoder(handle_unknown='use_encoded_value', 
                                     unknown_value=np.nan))])
        transformers.append(('cat', categorical_transformer, cat_features))
    if len(transformers): 
        preprocessor = ColumnTransformer(transformers=transformers)
        pipeline.append(('preprocessor', preprocessor))
    if 'smote' in model_str:
        smt = SMOTE(random_state=random_state)
        pipeline.append(('smt', smt))
    return pipeline


# def objective(
#         trial, train, test, target, model_str, 
#         random_state, dataset_filename, cat_features=['wettability']):
#     params = get_params(trial, model_str, random_state, cat_features)
#     model = get_model(model_str, params)
#     preproc_pipeline = get_preproc_pipeline(model_str=model_str, random_state=random_state, 
#                                             cat_features=cat_features, 
#                                             num_features=dict_num_features[dataset_filename])
#     pipeline = [('model', model)]
#     if preproc_pipeline: pipeline.insert(0, preproc_pipeline[0])
#     pipeline = Pipeline(steps=pipeline)
#     pipeline.fit(train.drop(columns=[target]), train[target])
#     preds = pipeline.predict(test.drop(columns=[target]))
#     f1 = f1_score(test[target], preds)
#     return f1
