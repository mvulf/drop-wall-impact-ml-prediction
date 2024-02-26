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


def _create_pipeline(*,
                    numerical_features, 
                    categorical_features,
                    model,
                    random_state,
                    cat_features_processor='onehot'):
    features_to_leave = []
    num_features = list(set(numerical_features) - set(features_to_leave))
    cat_features = list(set(categorical_features) - set(features_to_leave))
    transformers = []
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    transformers.append(
        ("num", numeric_transformer, num_features))
    if categorical_features is not None:
        dict_transformer = {
            'onehot': OneHotEncoder(handle_unknown="ignore"),
            'ordenc': OrdinalEncoder(handle_unknown='ignore')}
        # categorical_transformer = Pipeline(
        #     steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        categorical_transformer = Pipeline(
            steps=[(cat_features_processor, dict_transformer[cat_features_processor])])
        transformers.append(('cat', categorical_transformer, cat_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    smt = SMOTE(random_state=random_state)
    clf = Pipeline([("preprocessor", preprocessor), ("smt", smt), ("model", model)])
    return clf


class SklearnModelsPipeline:
    def __init__(self, train, test, target, model,
                 numerical_features, categorical_features,
                 random_state,
                 postfix='',
                 features_to_leave=[]):
        self.train = train
        self.test = test
        self.target = target
        self.model = model
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.features_to_leave = features_to_leave
        self.clf = _create_pipeline(
            numerical_features=numerical_features, 
            categorical_features=categorical_features,
            model=model,
            random_state=random_state)
        self.filename = ''
        model_name = str(self.model.__class__).split('.')[-1][:-2]
        self.model_name = f'{model_name.lower()}_{self.target}'
        if len(postfix): self.model_name = f'{self.model_name}_{postfix}'
        self.path_models = Path('modelling2_models')


    def _fit(self):
        self.clf.fit(
            X=self.train.drop(
                columns=[self.target]), 
                y=self.train[self.target])


    def _predict(self):
        return self.clf.predict(X=self.test.drop(
            columns=[self.target]))
    

    def fit_predict(self):
        self._fit()
        preds = self._predict()
        return preds


    def calculate_metrics(self):
        y_true = self.test[self.target]
        preds = self._predict()
        df_metrics = pd.DataFrame({
            'model': [self.model_name],
            'accuracy': [accuracy_score(y_true, preds)],
            'f1': [f1_score(y_true, preds)],
            'precision': [precision_score(y_true, preds)],
            'recall': [recall_score(y_true, preds)],
            'roc_auc': [roc_auc_score(y_true, preds)]})
        return df_metrics
    

    def _save_model(self):
        if not os.path.exists(self.path_models): os.makedirs(self.path_models)
        joblib.dump(self.clf, self.path_models / self.model_name)


    def full_pipeline(self, save_model=True):
        self._fit()
        df_metrics = self.calculate_metrics()
        if not save_model: return None
        self.save_model()
        print(f'{self.filename} was saved in {str(self.path_models)}')