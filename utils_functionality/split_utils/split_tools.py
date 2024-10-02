from pathlib import Path
import pandas as pd


def get_train_test(
    *, 
    dataset_filename:str, 
    target:str,
    path_data=Path('../data/before_drag_force_update'), # Path('../data')
    target_set={'splashing', 'net_impact'}, # {'splashing', 'no_fragmentation'}
):
    """
    Create train/test datasets.
    * dataset_filename - dataset for modelling
    * target - target for ML
    """
    df_indexes = pd.read_excel(
        path_data / f'df_ml_split_{target}.xlsx', index_col=[0])
    df_data = pd.read_excel(
        path_data / f'{dataset_filename}.xlsx')
    df_data.drop(
        columns=target_set-{target}, 
        inplace=True
    )
    train = df_data.loc[df_indexes.loc[df_indexes['sample']=='train', 'index']]
    test = df_data.loc[df_indexes.loc[df_indexes['sample']!='train', 'index']]
    return train, test


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