from featurewiz import featurewiz
import pandas as pd


class FeatureWizModel:
    def __init__(self, df, target, verbose):
        self.target = target
        self.df = df.copy()
        self.verbose = verbose

    def feature_selection(self, corr_limit=0.7):
        self.feats, self.df_featurewiz = featurewiz(
            self.df, self.target, corr_limit=corr_limit, verbose=self.verbose
        )


class CreateSamples:
    def __init__(
        self,
        df,
        train,
        test,
        target,
        features=None,
        dummies=False,
        use_featurewiz=False,
        corr_limit=0.7,
        strange_columns=None,
        drop_duplicates=False,
        verbose=0,
    ):
        self.train, self.test = train.copy(), test.copy()
        self.target_columns = [
            "splashing",
            "breaking_up",
            "net_impact",
            "rebound",
            "one_drop",
        ]
        self.target = target
        self.features = features
        drop_columns = list(set(self.target_columns) - set([target])) + ["test"]
        if strange_columns is not None:
            drop_columns += strange_columns
        obj_columns = df.select_dtypes(include=["object"]).columns
        if dummies:
            self.df = pd.get_dummies(df, columns=obj_columns)[features + [target]]
        else:
            self.df = df.drop(columns=drop_columns, axis=1).copy()
        if use_featurewiz:
            fwiz = FeatureWizModel(df=self.df, verbose=verbose, target=target)
            fwiz.feature_selection(corr_limit)
            self.df, self.features = fwiz.df_featurewiz, fwiz.feats
        self.y = self.df[self.target]
        if self.features is not None:
            to_drop = set(self.df.columns) - set(self.features)
            self.X = self.df.drop(to_drop, axis=1)
        else:
            self.X = self.df.drop(target, axis=1)
        self.X_train, self.y_train = (
            self.X.loc[self.train.index],
            self.y.loc[self.train.index],
        )
        self.X_test, self.y_test = (
            self.X.loc[self.test.index],
            self.y.loc[self.test.index],
        )
        if drop_duplicates:
            self.X_train = self.X_train.drop_duplicates()
            self.X_test = self.X_test.drop_duplicates()
            self.y_train = self.y_train.loc[self.X_train.index]
            self.y_test = self.y_test.loc[self.X_test.index]
            print(f"Shape train:\t {self.X_train.shape}")
            print(f"Shape test:\t {self.X_test.shape}")

    def get_samples(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
