from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np


def _create_pipeline(*,
                    numerical_features, 
                    categorical_features,
                    model,
                    random_state,
                    smote,
                    cat_features_processor='onehot',
                     features_to_leave = []):
   
    num_features = list(set(numerical_features) - set(features_to_leave))
    cat_features = list(set(categorical_features) - set(features_to_leave))
    smt = SMOTE(random_state=random_state)
    pipeline = [("model", model)]
    if ('xgboost' in str(model.__class__)) or ('catboost' in str(model.__class__)):
        if smote: pipeline.insert(0, ('smt', smt))
        return Pipeline(pipeline)
    transformers = []
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    transformers.append(
        ("num", numeric_transformer, num_features))
    if categorical_features is not None:
        dict_transformer = {
            'onehot': OneHotEncoder(handle_unknown="ignore"),
            'ordenc': OrdinalEncoder(handle_unknown='use_encoded_value', 
                                     unknown_value=np.nan)}

        categorical_transformer = Pipeline(
            steps=[(cat_features_processor, dict_transformer[cat_features_processor])])
        transformers.append(('cat', categorical_transformer, cat_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    pipeline = [("preprocessor", preprocessor), ("model", model)]
    if smote: pipeline.insert(1, ("smt", smt))
    return Pipeline(pipeline)


class MLPipeline:
    def __init__(self, train, test, target, model,
                 numerical_features, categorical_features,
                 random_state, dataset_filename,
                 smote=True, postfix='onehot',
                 features_to_leave=[]):
        self.train = train
        self.test = test
        self.target = target
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.features_to_leave = features_to_leave
        self.filename = ''
        self.dataset_filename = dataset_filename
        model_name = str(self.model.__class__).split('.')[-1][:-2]
        self.model_name = f'{model_name.lower()}_smote_{self.target}_{dataset_filename}'
        if not smote: self.model_name = self.model_name.replace('_smote', '')
        if len(postfix): self.model_name = f'{self.model_name}_{postfix}'
        self.clf = _create_pipeline(
            numerical_features=numerical_features, 
            categorical_features=categorical_features,
            model=model, cat_features_processor=postfix,
            smote=smote, random_state=random_state)
        self.path_models = Path('../modelling2_models')


    def _fit(self):
        if 'catboost' in str(self.model.__class__):
            self.clf.fit(
                X=self.train.drop(
                    columns=[self.target]), 
                    y=self.train[self.target],
                    model__cat_features=self.categorical_features)
        else: 
            self.clf.fit(X=self.train.drop(
                    columns=[self.target]), 
                    y=self.train[self.target])


    def _predict(self):
        return self.clf.predict(self.test.drop(
            columns=[self.target]))
    

    def fit_predict(self):
        self._fit()
        preds = self._predict()
        return preds


    def _calculate_metrics(self):
        y_true = self.test[self.target]
        preds = self._predict()
        df_metrics = pd.DataFrame({
            'dataset': [self.dataset_filename],
            'target': [self.target],
            'model': [self.model_name.replace(f'_{self.dataset_filename}', '')],
            'accuracy': [accuracy_score(y_true, preds)],
            'f1': [f1_score(y_true, preds)],
            'precision': [precision_score(y_true, preds)],
            'recall': [recall_score(y_true, preds)],
            'roc_auc': [roc_auc_score(y_true, preds)]})
        return df_metrics
    

    def _save_metrics(self, df_metrics):
        filename_results = '../results/metrics_modelling2.xlsx'
        if os.path.isfile(filename_results):
            existing_df = pd.read_excel(filename_results)
            combined_df = pd.concat((existing_df, df_metrics), ignore_index=True)
            combined_df.drop_duplicates(inplace=True)
            combined_df.to_excel(filename_results, index=False)
        else:
            df_metrics.to_excel(filename_results, index=False)


    def _save_model(self):
        if not os.path.exists(self.path_models): os.makedirs(self.path_models)
        joblib.dump(self.clf, self.path_models / self.model_name)


    def full_pipeline(self, save_model=True):
        self._fit()
        df_metrics = self._calculate_metrics()
        self._save_metrics(df_metrics)
        if not save_model: return None
        self._save_model()
        print(f'{self.model_name} was saved in {str(self.path_models)}')