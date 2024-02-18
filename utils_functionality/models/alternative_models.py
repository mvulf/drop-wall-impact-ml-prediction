import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_pipeline(numeric_features, model, categorical_features=None):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    transformers = []
    transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features is not None:
        categorical_transformer = Pipeline(
            steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformers.append(categorical_transformer)
    preprocessor = ColumnTransformer(transformers=transformers)
    smt = SMOTE(random_state=42)
    clf = Pipeline([("preprocessor", preprocessor), ("smt", smt), ("model", model)])
    return clf
