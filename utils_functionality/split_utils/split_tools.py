from pathlib import Path
import pandas as pd


def get_train_test(
    *, 
    target:str,
    df=None,
    dataset_filename:str=None, 
    path_data=Path('../data/before_drag_force_update'), # Path('../data')
    target_set={'splashing', 'net_impact'}, # {'splashing', 'no_fragmentation'}
    verbose=True,
):
    """
    Create train/test datasets.
    * dataset_filename - dataset for modelling
    * target - target for ML
    """
    
    if df is None:
        if verbose:
            print('No df was gotten.')
        df=load_df(
            dataset_filename=dataset_filename,
            path_data=path_data,
            target=target,
            target_set=target_set,
            verbose=verbose,
        )
    
    index_path = path_data / f'df_ml_split_{target}.xlsx'
    if verbose:
        print(f'Load split indexes from: {index_path}')
    
    df_indexes = pd.read_excel(
        index_path, 
        index_col=[0]
    )
    train = df.loc[df_indexes.loc[df_indexes['sample']=='train', 'index']]
    test = df.loc[df_indexes.loc[df_indexes['sample']!='train', 'index']]
    return train, test


def load_df(
    *,
    dataset_filename:str,
    path_data,
    target:str,
    target_set:set,
    verbose=True,
):
    df_path = path_data / f'{dataset_filename}.xlsx'
    df = pd.read_excel(
        df_path
    )
    df.drop(
        columns=target_set-{target}, 
        inplace=True
    )
    if verbose:
        print(f'Load dataset from: {df_path}')
        print(f'Keep "{target}" from {target_set}')
    
    return df


# WITH OLD NAME "NET_IMPACT"
# def get_train_test(*, dataset_filename: str, target: str):
#     """
#     Create train/test datasets.
#     * dataset_filename - dataset for modelling
#     * target - target for ML
#     """
#     path_data = Path('../data/before_drag_force_update')
#     df_indexes = pd.read_excel(
#         path_data / f'df_ml_split_{target}.xlsx', index_col=[0])
#     df_data = pd.read_excel(
#         path_data / f'{dataset_filename}.xlsx')
#     df_data.drop(columns={'splashing', 'net_impact'}-{target}, inplace=True)
#     train = df_data.loc[df_indexes.loc[df_indexes['sample']=='train', 'index']]
#     test = df_data.loc[df_indexes.loc[df_indexes['sample']!='train', 'index']]
#     return train, test