# Notebooks with analysis

## EDA update.ipynb
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

## feature_engineering.ipynb
Used to combine '../data/data_target.xlsx', '../data/suspension_experiments.xlsx', '../data/Substrates.xlsx' and create **'../data/df_merged_edited.xlsx'** and report 'report_edited.html'.

Then **'../data/df_merged_edited.xlsx'** is analysed

## research.ipynb
Study of the final dataset **'../data/df_main.xlsx'** to obtain important experimental results
