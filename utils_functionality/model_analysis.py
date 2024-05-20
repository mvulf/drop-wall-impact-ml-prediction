import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import seaborn as sns
sns.set_style('darkgrid')

from IPython.display import display

import os
from pathlib import Path
import joblib



"""________DATA PREPARATION__________
"""

def get_volume_fraction_type(value):
    if value == 1:
        return '0.08 .. 0.10'
    else:
        return '0.04 .. 0.05'
    

def get_text_definitions(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    
    # Prepare text definitions of the features
    df['net_impact_type'] = df['net_impact'].map(
        net_impact_type_df['impact_type'].to_dict()
    )
    df['splashing_type'] = df['splashing_spectrum'].map(
        splash_impact_type_df
        .sort_values('value')['impact_type']
        .reset_index(drop=True)
        .to_dict()
    )
    
    return df


def create_mesh_df(**params_for_mesh):
    # Get meshed values
    value_list = list(params_for_mesh.values())
    values = np.meshgrid(*value_list)
    # Get array of values
    stacked_array = np.hstack([value.reshape(-1, 1) for value in values])

    df = pd.DataFrame(
        stacked_array,
        columns = params_for_mesh.keys()
    )
    return df


def get_top_object_value(series:pd.Series):
    return series.value_counts().sort_values(ascending=False).index[0]


def get_const_params(
    df:pd.DataFrame,
    columns:list,
):
    params_dict = {}
    
    for column in columns:
        series = df[column]
        # For object - get frequent value
        if series.dtype == 'O':
            params_dict[column] = get_top_object_value(series)
        # For other types
        else:
            value = series.median()
            if series.dtype == 'int64':
                value = int(value)
            params_dict[column] = value
            
    return params_dict


def create_dataframe(
    const_params:dict,
    variable_df:pd.DataFrame,
):
    df = variable_df.copy()
    
    for column in const_params:
        df[column] = const_params[column]
    return df


def extract_agg_features(source_df):
    df = source_df.copy()
    
    Re_numerator = df['velocity'] * df['droplet_diameter'] * df['liquid_density']
    
    df['Re'] = Re_numerator/df['viscosity']
    df['We'] = df['velocity'] * Re_numerator / df['surface_tension']
    
    df['We_Re'] = df['We']**(1/2) * df['Re']**(1/4)
    
    return df


def get_poly_df(source_df):
    
    # To be sure, that source_df does not change
    df = source_df.copy()
    
    df['Re^(1/4)'] = df['Re']**(1/4)
    df['We^(1/4)^2'] = df['We']**(1/2)
    
    df['We^(1/4)^2 Re^(1/4)'] = df['We^(1/4)^2'] * df['Re^(1/4)']
    
    return df


def extract_features(source_df):
    df = source_df.copy()
    df = extract_agg_features(df)
    df = get_poly_df(df)
    
    return df


"""______PREDICTIONS_______
"""


def predict_proba(df, model, model_features):
    proba = model.predict_proba(
        df[model_features]
    )[:,1] # Probability of 1-class
    
    return proba


def predict_all_proba(source_df, model_list):
    
    df = source_df.copy()
    
    for model_name, model, model_features in model_list:
        df[model_name] = predict_proba(df, model, model_features)
    
    return df


def get_contour_df(
    df_model,
    net_impact_model_features,
    splashing_model_features,
    velocity:np.ndarray=np.linspace(0.0, 7.0, 50),
    particle_liquid_density_ratio:np.ndarray=np.linspace(0.3, 1.9, 50),
    particle_mean_diameter:np.ndarray=np.linspace(20e-6, 350e-6, 50),
    const_params:list=[
        'wettability',
        'inclination',
        'roughness_binary',
        'volume_fraction_binary',
        'droplet_diameter',
        'liquid_density',
        'surface_tension',
        'viscosity',
    ],
    density_params:list=[
        'particle_liquid_density_ratio'
    ],
    diameter_params:list=[
        'particle_droplet_diameter_ratio', 
    ],
    dynamic_params:list=[
        'velocity', 
        'Re', 
        'We', 
        'We_Re'
    ],
    verbose:bool=True,
):
    df_model = df_model.copy()

    vel_dens_df = create_mesh_df(
        velocity=velocity,
        particle_liquid_density_ratio=particle_liquid_density_ratio
    )

    vel_diam_df = create_mesh_df(
        velocity=velocity,
        particle_mean_diameter=particle_mean_diameter,
    )

    const_params_list_dens = const_params.copy()
    const_params_list_dens.extend(diameter_params)
    const_params_list_diam = const_params.copy()
    const_params_list_diam.extend(density_params)

    if verbose:
        print('vel_dens_df')
        display(vel_dens_df)
        print('vel_diam_df')
        display(vel_diam_df)
        print('const_params_list_dens')
        display(const_params_list_dens)
        print('const_params_list_diam')
        display(const_params_list_diam)

    dens_pred_df = create_dataframe(
        const_params=get_const_params(
            df_model, 
            const_params_list_dens
        ),
        variable_df=vel_dens_df
    )

    diam_pred_df = create_dataframe(
        const_params=get_const_params(
            df_model, 
            const_params_list_diam
        ),
        variable_df=vel_diam_df
    )
    diam_pred_df['particle_droplet_diameter_ratio'] = (
        diam_pred_df['particle_mean_diameter']
        / df_model['droplet_diameter'].median()
    )
    diam_pred_df = diam_pred_df.drop('particle_mean_diameter', axis=1)

    if set(diam_pred_df.columns) == set(dens_pred_df.columns):
        print('Dataframes are equal')
    else:
        print('WARNING: Dataframes are NOT equal')

    dens_pred_df = extract_features(dens_pred_df)
    diam_pred_df = extract_features(diam_pred_df)

    if verbose:
        print('dens_pred_df after We, Re extraction')
        dens_pred_df.info()
        print('diam_pred_df after We, Re extraction')
        diam_pred_df.info()

    net_impact_columns = set(net_impact_model_features)
    extra_columns = net_impact_columns - set(dens_pred_df.columns)
    if len(extra_columns)>0:
        print('Net-impact: columns for additional creation')
        display(extra_columns)
    else:
        print('Net-impact: No columns are needed for creation')

    splashing_columns = set(splashing_model_features)
    extra_columns = splashing_columns - set(dens_pred_df.columns)
    if len(extra_columns)>0:
        print('Splashing: columns for additional creation')
        display(extra_columns)
    else:
        print('Splashing: No columns are needed for creation')
    
    print('Dataframes are prepared')
    
    return dens_pred_df, diam_pred_df


"""________MODELS__________
"""

def get_best_model_name(sorted_df, extention='.pkl'):
    if len(sorted_df) > 0:
        best_model = sorted_df.iloc[0]
        
        if 'svc' in best_model['model']:
            model_name = '_'.join(best_model['model'].split('_')[:-1])
            svc_type = best_model['model'].split('_')[-1]

            name = (
                    '_'.join([model_name, best_model['dataset'], svc_type]) 
                    + extention
                )
        else:
            name = (
                    '_'.join(best_model[['model', 'dataset']].values) 
                    + extention
                )
        return name


def get_best_model_name_no_opt(sorted_df, extention=''):
    if len(sorted_df) > 0:
        best_model = sorted_df.iloc[0]
        
        if 'ordenc' in best_model['model'] or 'onehot' in best_model['model']:
            model_name = '_'.join(best_model['model'].split('_')[:-1])
            enc_type = best_model['model'].split('_')[-1]

            name = (
                    '_'.join([model_name, best_model['dataset'], enc_type]) 
                    + extention
                )
        else:
            name = (
                    '_'.join(best_model[['model', 'dataset']].values) 
                    + extention
                )
        return name
    

def get_targer_metrics_dict(metrics_df):
    
    net_impact_models = metrics_df[metrics_df['target'] == 'net_impact']
    splashing_models = metrics_df[metrics_df['target'] == 'splashing']

    metrics_dict = {
        'splashing': splashing_models,
        'net_impact': net_impact_models
    }
    
    return metrics_dict


def get_best_models(
    target_metrics_dict,
    models_type_list,
    verbose=True,
    order=['f1', 'roc_auc', 'optuna_flg'],
    extention='.pkl',
):
    best_models_name = {}
    
    for target, model_df in target_metrics_dict.items():
        # best_models_name[target] = {}
        print()
        print(f'TARGET: {target.upper()}, models count {model_df.shape[0]}')
        
        for model_type in models_type_list:
            print(model_type)
            
            dimless_model_df = (
                model_df[model_df['dataset'].str.contains('dimensionless')]
            )
            
            sub_model_df = (
                dimless_model_df[dimless_model_df['model']
                .str.contains(model_type)]
                .sort_values(order, ascending=False)
            )
            
            if len(sub_model_df) == 0:
                sub_model_df = (
                    model_df[model_df['model']
                    .str.contains(model_type)]
                    .sort_values(
                        ['f1', 'roc_auc', 'optuna_flg'], 
                        ascending=False
                    )
                )
            
            best_models_name.setdefault(model_type, {})
            best_models_name[model_type][target] = (
                get_best_model_name(sub_model_df, extention=extention)
            )
            if verbose:
                display(sub_model_df)

    return best_models_name


def get_best_models_no_opt(
    target_metrics_dict,
    models_type_list,
    verbose=True,
    order=['f1', 'roc_auc', 'optuna_flg'],
    extention='',
):
    best_models_name = {}
    
    for target, model_df in target_metrics_dict.items():
        # best_models_name[target] = {}
        print()
        print(f'TARGET: {target.upper()}, models count {model_df.shape[0]}')
        
        for model_type in models_type_list:
            print(model_type)
            
            dimless_model_df = (
                model_df[model_df['dataset'].str.contains('dimensionless')]
            )
            
            sub_model_df = (
                dimless_model_df[dimless_model_df['model']
                .str.contains(model_type)]
                .sort_values(order, ascending=False)
            )
            
            if len(sub_model_df) == 0:
                sub_model_df = (
                    model_df[model_df['model']
                    .str.contains(model_type)]
                    .sort_values(
                        ['f1', 'roc_auc', 'optuna_flg'], 
                        ascending=False
                    )
                )
            
            best_models_name.setdefault(model_type, {})
            best_models_name[model_type][target] = (
                get_best_model_name_no_opt(sub_model_df, extention=extention)
            )
            if verbose:
                display(sub_model_df)

    return best_models_name

"""________PLOTS__________
"""

# Preprare impact types
splash_impact_type_dt = {
    'impact_type': ['no splashing', 'splashing', 'semi splashing',],
    'value': [0, 2, 1,],
    'color': ['b', 'r', 'g',],
    'marker': ['x', 'o', 's',],
    'order': [0, 2, 1,] # legend order
}

splash_impact_type_df = pd.DataFrame(splash_impact_type_dt)
display(splash_impact_type_df)

net_impact_type_dt = {
    'impact_type': ['unclear impact', 'net impact'],
    'value': [0, 1],
    'color': ['r', 'b'],
    'marker': ['x', 'o'],
    'order': [0, 1] # legend order
}

net_impact_type_df = pd.DataFrame(net_impact_type_dt)
display(net_impact_type_df)


def plot_WeRe_scatter(
    scatter_df:pd.DataFrame,
    impact_type_name:str,
    impact_type_df:pd.DataFrame,
    y_feature_name:str,
    y_label:str,
    ax:mpl.axes.Axes,
    markersize:int=15,
):
    # Plot scatters
    for i, row in impact_type_df.iterrows():
        impact_group = scatter_df[scatter_df[impact_type_name] == row['value']]
        
        ax.scatter(
            x=impact_group['We_Re'],
            y=impact_group[y_feature_name],
            marker=row['marker'],
            s=markersize,
            color=row['color'],
            label=row['impact_type'],
        )
        
    #get handles and labels
    handles, labels = ax.get_legend_handles_labels()

    #specify order of items in legend
    order = impact_type_df['order'].values

    #add legend to plot
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 

    ax.set_xlabel('$We^{1/2} Re^{1/4}$')

    ax.set_ylabel(y_label);
    
    return ax


def plot_WeRe_contour_scatter(
    contour_df:pd.DataFrame,
    scatter_df:pd.DataFrame,
    impact_type_name:str,
    impact_type_df:pd.DataFrame,
    y_feature_name:str,
    y_label:str,
    ax:mpl.axes.Axes,
    cmap_fill='coolwarm',
    cmap_contour='Greys',
    levels_fill=50,
    levels_contour=[0.3, 0.4, 0.5, 0.6, 0.7],
    markersize:int=15,
    fontsize:int=12,
    contour_labels:bool=True,
):
    # Mesh Values
    x = contour_df['We_Re'].unique()
    y = contour_df[y_feature_name].unique()
    probas = contour_df[impact_type_name].values.reshape(x.size, y.size)
    
    contourfplot = ax.contourf(
        x,
        y,
        probas,
        cmap=cmap_fill,
        levels=levels_fill,
        vmin=0.,
        vmax=1.,
        alpha=0.8,
        antialiased=True,
    )
    plt.colorbar(contourfplot, ax=ax)
    
    contplot = ax.contour(
        x,
        y,
        probas,
        cmap=cmap_contour,
        levels=levels_contour,
        vmin=0.,
        vmax=1.,
    )
    #contour line labels
    if contour_labels:
        ax.clabel(contplot, fmt = '%1.1f', colors = 'k', fontsize=fontsize)
    
    ax = plot_WeRe_scatter(
        scatter_df,
        impact_type_name,
        impact_type_df,
        y_feature_name,
        y_label,
        ax,
        markersize,
    )
    
    ax.grid(color='black', alpha=0.5)
    
    return ax


def plot_all_WeRe_scatters(df):

    y_feature_name = 'particle_liquid_density_ratio'
    y_label = '$\\rho_{p}/\\rho_{l}$'

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=600)

    plot_WeRe_scatter(
        scatter_df=df,
        impact_type_name='splashing_spectrum',
        impact_type_df=splash_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        ax=axes[0]
    )

    plot_WeRe_scatter(
        scatter_df=df,
        impact_type_name='net_impact',
        impact_type_df=net_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        ax=axes[1]
    );

    axes[0].set_title('Splashing');
    axes[1].set_title('Net impact');


    # PARTICLE DROPLET DIAMETER RATIO

    y_feature_name = 'particle_droplet_diameter_ratio'
    y_label = '$d_p/D_{drop}$'

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=600)

    plot_WeRe_scatter(
        scatter_df=df,
        impact_type_name='splashing_spectrum',
        impact_type_df=splash_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        ax=axes[0]
    )

    plot_WeRe_scatter(
        scatter_df=df,
        impact_type_name='net_impact',
        impact_type_df=net_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        ax=axes[1]
    );

    axes[0].set_title('Splashing');
    axes[1].set_title('Net impact');

    # PHI-PLOT

    sns.color_palette('Set2')

    fig, (ax_splash, ax_net_impact) = plt.subplots(1, 2, figsize=(12,8))

    sns.swarmplot(
    data=df, 
    x='We_Re', 
    y='volume_fraction_type', 
    hue='splashing_type', # style='splashing_type', 
    ax=ax_splash, 
    orient='h', 
    size=4.95, 
    order=['0.08 .. 0.10', '0.04 .. 0.05'],
    hue_order=['no splashing', 'semi splashing', 'splashing'],
    palette=list(splash_impact_type_df.sort_values('value')['color'].values)
    )

    sns.swarmplot(
    data=df, 
    x='We_Re', 
    y='volume_fraction_type', 
    hue='net_impact_type', # style='splashing_type', 
    ax=ax_net_impact, 
    orient='h', 
    size=4.95, 
    order=['0.08 .. 0.10', '0.04 .. 0.05'],
    palette=list(net_impact_type_df['color'].values)
    )

    axes = (ax_splash, ax_net_impact)

    for ax in axes:
        # ax.set_xticks(list(np.arange(0, 510, 50)));
        ax.set_xlabel('$We^{1/2} Re^{1/4}$')

    ax_net_impact.get_yaxis().set_visible(False)
    ax_splash.set_ylabel('$\phi$');

    ax_splash.set_title('Splashing');
    ax_net_impact.set_title('Net impact');

    fig.tight_layout()
    

def plot_final_plots(
    dens_pred_df_res,
    diam_pred_df_res,
    scatter_df,
    model_name,
    splashing_levels=[0.3, 0.4, 0.5, 0.6, 0.7],
    net_impact_levels=[0.5, 0.6, 0.7, 0.8],
    splashing_levels_fill=50,
    net_impact_levels_fill=50,
    splashing_contour_labels=True,
    net_impact_contour_labels=True,
    figsize=(12, 12),
):
    y_feature_name = 'particle_liquid_density_ratio'
    y_label = '$\\rho_{p}/\\rho_{l}$'

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=600)

    plot_WeRe_contour_scatter(
        contour_df=dens_pred_df_res,
        scatter_df=scatter_df,
        impact_type_name='splashing',
        impact_type_df=splash_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        levels_fill=splashing_levels_fill,
        levels_contour=splashing_levels,
        fontsize=10,
        contour_labels=splashing_contour_labels,
        ax=axes[0,0]
    )

    plot_WeRe_contour_scatter(
        contour_df=dens_pred_df_res,
        scatter_df=scatter_df,
        impact_type_name='net_impact',
        impact_type_df=net_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        cmap_fill='RdBu',
        levels_fill=net_impact_levels_fill,
        levels_contour=net_impact_levels,
        fontsize=10,
        contour_labels=net_impact_contour_labels,
        ax=axes[0,1]
    );

    axes[0,0].set_title('Splashing classification on density');
    axes[0,1].set_title('Net impact classification on density');
    
    y_feature_name = 'particle_droplet_diameter_ratio'
    y_label = '$d_p/D_{drop}$'

    plot_WeRe_contour_scatter(
        contour_df=diam_pred_df_res,
        scatter_df=scatter_df,
        impact_type_name='splashing',
        impact_type_df=splash_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        levels_fill=splashing_levels_fill,
        levels_contour=splashing_levels,
        fontsize=10,
        contour_labels=splashing_contour_labels,
        ax=axes[1,0]
    )

    plot_WeRe_contour_scatter(
        contour_df=diam_pred_df_res,
        scatter_df=scatter_df,
        impact_type_name='net_impact',
        impact_type_df=net_impact_type_df,
        y_feature_name=y_feature_name,
        y_label=y_label,
        cmap_fill='RdBu',
        levels_fill=net_impact_levels_fill,
        levels_contour=net_impact_levels,
        fontsize=10,
        contour_labels=net_impact_contour_labels,
        ax=axes[1,1]
    );

    # Change y-axis
    for ax in axes[1,:].flat:
        ax.set_yticks(np.arange(0.01, 0.12, 0.02))
        
    axes[1,0].set_title('Splashing classification on diameter');
    axes[1,1].set_title('Net impact classification on diameter');
    
    fig.suptitle(model_name)
    fig.tight_layout()


def plot_all_final_plots(
    best_models_name,
    df_model,
    scatter_df,
    models_folder=Path('..', 'results', 'best_models_modelling_2'),
    save_plots=False,
    save_prefix='',
    save_path=Path('..', 'results')
):
    
    for model_name in best_models_name:
        model = {}
        model_features = {}
        for target in best_models_name[model_name]:
            full_model_name = best_models_name[model_name][target]
            model_path = Path(models_folder, full_model_name)
            if not os.path.isfile(model_path):
                print(f"ERROR: {full_model_name}")
            model[target] = joblib.load(model_path)
            display(model_path)
            display(model[target])
            
            if 'catboostclassifier' in model_name:
                model_features[target] = model[target][0].feature_names_
            else:
                model_features[target] = model[target][0].feature_names_in_
            display(model_features[target])
        
        # Prepare dataframes
        net_impact_model_features = model_features['net_impact']
        splashing_model_features = model_features['splashing']
        
        dens_pred_df, diam_pred_df = get_contour_df(
            df_model=df_model,
            net_impact_model_features=net_impact_model_features,
            splashing_model_features=splashing_model_features,
            verbose=False,
        )
        
        # Predictions and plot
        model_list = [
            ('splashing', model['splashing'], splashing_model_features),
            ('net_impact', model['net_impact'], net_impact_model_features)
        ]

        dens_pred_df_res = predict_all_proba(dens_pred_df, model_list)
        diam_pred_df_res = predict_all_proba(diam_pred_df, model_list)

        plot_final_plots(
            dens_pred_df_res=dens_pred_df_res,
            diam_pred_df_res=diam_pred_df_res,
            scatter_df=scatter_df,
            # splashing_levels=[0.3],
            # net_impact_levels=[0.5],
            splashing_levels_fill=50,
            net_impact_levels_fill=50,
            model_name=model_name
        )
        if save_plots:
            plt.savefig(
                Path(
                    save_path, 
                    f'{save_prefix}{model_name}_WeRe_classification.pdf',
                ),
                dpi=600
            )