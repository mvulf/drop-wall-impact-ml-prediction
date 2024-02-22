from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import uuid
import joblib


def _create_pipeline(*,
                     numerical_features, 
                     categorical_features,
                     model,
                     random_state):
    features_to_leave = []
    num_features = list(set(numerical_features) - set(features_to_leave))
    cat_features = list(set(categorical_features) - set(features_to_leave))
    transformers = []
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    transformers.append(
        ("num", numeric_transformer, num_features))
    if categorical_features is not None:
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
        transformers.append(('cat', categorical_transformer, cat_features))
    preprocessor = ColumnTransformer(transformers=transformers)
    smt = SMOTE(random_state=random_state)
    clf = Pipeline([("preprocessor", preprocessor), ("smt", smt), ("model", model)])
    return clf


class SklearnModelsPipeline:
    def __init__(self, train, test, target, model,
                 numerical_features, categorical_features,
                 random_state,
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
    

    def _generate_filename(self):
        id_ = uuid.uuid4().__str__().split('-')[0]
        model_name = str(self.model.__class__).split('.')[-1][:-2]
        self.filename = f'{model_name}_{id_}.pkl'
        return self.filename
    

    def save_model(self):
        filename = self.filename
        joblib.dump(self.clf, self.filename)

    # def full_pipeline(self):

        