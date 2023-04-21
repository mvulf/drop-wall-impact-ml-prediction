from IPython.display import display, HTML
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport


def _display_target_distr(train, test, target_columns):
    for target in target_columns:
        df_train = pd.DataFrame(train[target].value_counts())
        df_test = pd.DataFrame(test[target].value_counts())
        display(HTML(f'<h2>{target}</h2>'))
        df_train.rename(columns={target: 'train'}, inplace=True)
        df_test.rename(columns={target: 'test'}, inplace=True)
        df_concat = pd.concat((df_train, df_test), axis=1)
        df_train = pd.DataFrame(
            np.round(train[target].value_counts(normalize=True)*100, 4))
        df_test = pd.DataFrame(
            np.round(test[target].value_counts(normalize=True)*100, 4))
        df_train.rename(columns={target: 'train, %'}, inplace=True)
        df_test.rename(columns={target: 'test, %'}, inplace=True)
        df_train.rename(columns={target: 'train, %'}, inplace=True)
        df_test.rename(columns={target: 'test, %'}, inplace=True)
        df_concat = pd.concat((df_concat, df_train, df_test), axis=1)
        display(HTML(df_concat.to_html()))


def _create_profiler(df, train=True):
    path_var = 'test'
    title = 'Test EDA'
    if train:
        path_var = 'train'
        title = 'Train EDA'
    path = f'../data/report_{path_var}.html'
    profile = ProfileReport(df, title=title)
    print(f"Profiler creation: {path}")
    profile.to_file(path)


def get_class_results(
        train,
        test,
        target_columns=['splashing', 'breaking_up',
                        'net_impact', 'rebound', 'one_drop']
):
    display(HTML(f'<h2>Train shape:\t{train.shape}</h2>'))
    display(HTML(f'<h2>Test shape:\t{test.shape}</h2>'))
    display(HTML('<br><h1>Class distribution</h1>\n'))
    _display_target_distr(train, test, target_columns)
    _create_profiler(train)
    _create_profiler(test, train=False)
