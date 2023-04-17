from featurewiz import featurewiz


class FeatureWizModel:
    def __init__(self, df, target, target_columns, verbose):
        self.target = target
        self.df = df.copy()
        self.verbose = verbose

    def feature_selection(self, corr_limit=0.7):
        self.feats, self.df_featurewiz = featurewiz(
            self.df,
            self.target,
            corr_limit=0.7,
            verbose=self.verbose
        )


class CreateSamples:
    def __init__(self, df, train, test, target, features=None, use_featurewiz=False, corr_limit=0.7, verbose=0):
        self.target_columns = ['splashing',
                               'breaking_up', 'net_impact', 'rebound']
        self.target = target
        drop_columns = list(set(self.target_columns) -
                            set([target])) + ['test']
        self.df = df.drop(columns=drop_columns, axis=1).copy()
        self.features = features
        if use_featurewiz:
            fwiz = FeatureWizModel(
                df=self.df, verbose=verbose, target=target, target_columns=self.target_columns,
            )
            fwiz.feature_selection(corr_limit)
            self.df, self.features = fwiz.df_featurewiz, fwiz.feats

        self.y = self.df[self.target]
        if self.features is not None:
            to_drop = set(self.df.columns) - set(self.features)
            self.X = self.df.drop(to_drop, axis=1)
        else:
            self.X = self.df.drop(target, axis=1)
        self.X_train, self.y_train = self.X.loc[train.index], self.y[train.index]
        self.X_test, self.y_test = self.X.loc[test.index], self.y[test.index]

    def get_samples(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
