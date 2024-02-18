# Notebooks with analysis

This notebooks are introduced in the order to Run them:

1. **edit_labels.ipynb**: 
    
    - Load */data/archive/df_main_first_iteration.xlsx*, and */data/archive/Suspension Data Labeling Edited.xlsx* to find We_Re outliers and correct them!
    - After outlier analysis, **Suspension Data Labeling_We_Re_edit.xlsx** was created.
    - Create labels for:
        - targets = ['net_impact', 'splashing']
        - impact_labels = ['splashing_spectrum', 'breaking_up', 'rebound']
    - Prepare and save **df_main.xlsx**

2. **EDA_We_Re_edit.ipynb**:
    - Upload df_main.xlsx
    - Wettability to ordinal (0,1,2)
    - Drop unnecessary features
    - Check correlation of targets, labels, and features
    - Prepare datasets (see README in "data"-folder):
        - **df_modelling_full.xlsx**
        - **df_modelling_no_multicollinearity.xlsx**
        - **df_modelling_dimensionless.xlsx**

3. **research.ipynb**:
    - Upload df_main.xlsx
    - Drop unnecessary features
    - Prepare labels text definitions
    - Prepare visualization params
    - Plot Splashing and Net impact scatters in density_ratio - We_Re, diam_ratio - We_Re and phi - We_Re axis.
    - Plot boxplots (splashing, rebound, breaking up)
    - Check outliers (no outliers)
    - Collect "Glycerol data"
    - Plot "Glycerol data"
    - **ML-models prediction** (TODO: edit after creating new models)


# Unused notebooks:

- MODELLING: all first_iteration notebooks were saved in "archive":
    - modelling*.ipynb
    - tuning.ipynb
    - model_interpretation.ipynb

- archive/EDA update_firts_iteration.ipynb

    Used to find mistakes in '../data/Suspension Data Labeling.xlsx', replace them with correct values and save in '../data/Suspension Data Labeling Edited.xlsx'.
    Then '../data/Suspension Data Labeling Edited.xlsx' were analysed (including correlations), and dataset with final labels were created '../data/data_target.xlsx' (see "Формирование бинарных целевых признаков").
    - **Splashing**: when 'Number of detached small droplets during Spreading']=='many';
    - **Breaking up**: (df_data['Droplet Receding']==2) |
        (df_data['Number of detached small droplets during Spreading']=='few'). 
        
        This class combines clear "breaking_up" during receding and Spreading detaching;
        
    - **Rebound**: (df_data['Rebound']=='True') |
        (df_data['Rim merging or Central jet ejecting']==2).

        This class combines all rebound types (partial and total), as well as the central jet ejection!

    - **Net impact**: (df_data['Number of detached small droplets during Spreading']==0) &
        ((df_data['Droplet Receding']==0) | (df_data['Droplet Receding']==1)) &
        (df_data['Rebound']==0).

        Net impact - when there is **no Splashing** (no small droplets detached during spreading), **no Breaking up** [no small droplets detached during Receding or Rim merging (or the equivalent with Droplet Receding is introduced above)] and **no Rebound**. Thus 

- archive/feature_engineering_first_iteration.ipynb
    Used to combine '../data/data_target.xlsx', '../data/suspension_experiments.xlsx', '../data/Substrates.xlsx' and create **'../data/df_merged_edited.xlsx'** and report 'report_edited.html'.

    Then **'../data/df_merged_edited.xlsx'** is analysed

- archive/research_first_iteration.ipynb and research_preparation_firts_iteration.ipynb
Study of the final dataset **'../data/df_main.xlsx'** to obtain important experimental results. BEFORE EDITING ERRORS
