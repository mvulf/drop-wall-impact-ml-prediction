# Description

## Labels description.

In df_main.xlsx, df_modelling_*.ipynb

- **splashing_spectrum** (only in df_main.xlsx): 
    - **0 - no splashing**, when *'Number of detached small droplets during Spreading' == 0*
    - **1 - semi splashing**, when **gear** occurs, or when few droplets detach
    - **2 - splashing**. Pure splashing, when **many** droplets or **many small** droplets detach
    
- **splashing**: 1 - when *'Number of detached small droplets during Spreading' != 0*;
- **breaking_up** (only in df_main.xlsx): when *'Number of detached small droplets during Receding or Rim merging' > 0*
- **rebound** consists of next cases (only in df_main.xlsx):
    - **2 - true rebound**, when droplets detaching during partial rebound, or when droplet rebounds totally.
    
    *True rebound: 'Rebound' == 2 OR ('Rebound' == 1 AND 'Number of detached droplets during Rebound' > 0)*
    - **1 - jet ejection**, when true rebound does not appear.

    *'Rim merging or Central jet ejecting' == 2 AND NOT True rebound*    
    - **0** - no true rebound and no jet ejection. *Other cases*

- **net_impact**: when there is 
    - **no Splashing** (no small droplets detached during spreading, *'Number of detached small droplets during Spreading'==0*), 
    - **no Breaking up** (when *'Number of detached small droplets during Receding or Rim merging' == 0*)
    - **no True Rebound** (see True rebound definition early)


## Suspension Data Labeling_We_Re_edit.xlsx description

| Parameter name | Description | Possible values |
| --- | --- | --- |
| No satellites | 1 - if no satellites, 0 - satellites apper | 0, 1 |
| Number of detached small droplets during *spreading* | The number of small droplets that flew out of the lamella drops in the process of spreading. **Gear** - no droplets detaches, but shape looks like gear. **Many small** - when tiny droplets or even particles are detached from the droplet  | 0, gear, 1, .., 5, many small, many |
| Droplet Receding | Presence (1,2), absence — 0 of a visible drop shrinkage, and an increase in the drop height after reaching the maximum spot diameter. 1 — the usual reduction of the main drop, 2 — the destruction of the main drop in the process of reduction | 0, 1, 2 |
| Rim merging or Central jet ejecting | The presence (1,2), absence — 0 of an explicit rim merging at the end of the shrinkage of the drop (1) and/or the reverse vertical jet (2) (against gravity) | 0, 1, 2 |
| Number of detached small droplets during *Receding or Rim merging* | The number of small droplets separating from the rim of the main drop (rim) and remaining on the substrate during the shrinkage of the drop (receding). Drops do not separate — 0, several drops, many drops — many. **Particles** - particles are staying on the plate after droplet receding, but no large droplet detaching | 0, particles, 1, .., 5, many |
| Rebound | Presence (1,2), absence — 0 of a drop rebound after the formation of a reverse vertical jet (central jet ejecting). Partial rebound — 1, total rebound - 2 | 0, 1, 2 |
| Number of detached droplets during *Rebound* | The number of droplets separating from the reverse vertical jet (central jet) | 0, 1, .., 5, many |
| Final droplets count | The total number of droplets on the substrate, clearly separated from each other | 1, .., 5, many |


# Data Folder

## Essential dataframes and files:
- **df_main.xlsx** - main experimental dataset.
Previous dataset was *"archive/df_main_first_iteration.xlsx"* - old version of the main experimental dataset, which were used for the first iteration modelling (branch "modelling").
- **Suspension Data Labeling_We_Re_edit.xlsx** - dataset with correct preliminary labels. 
This dataset was used to create a final labels, such as 'splashing', 'breaking_up', 'net_impact', 'rebound'.
See *../notebooks/edit_labels.ipynb*.
Old version is: *"archive/Suspension Data Labeling Edited.xlsx"* (see *../notebooks/archive/EDA update_first_iteration.ipynb*).

FOR MODELLING:
- **df_modelling_full.xlsx** - dataframe whith dropped droplet generation parameters ('one_drop', 'voltage', 'long_impulse_duration', 'long_impulse_dur_binary'), test number ('test'), 'splashing_spectrum', 'breaking_up', 'rebound', 'height', 'particle_diameter_cat', 'volume_fraction'. However, there are still multicorrelated parameters. **It is not recommended for use**.
- **df_modelling_no_multicollinearity.xlsx** - df_modelling_full.xlsx with dropped 'particle_mean_diameter', 'particle_density', 'roughness', 'velocity' as highly correlated with 'particle_droplet_diameter_ratio', 'particle_liquid_density_ratio', 'roughness_binary', 'We' respectively. **Highly recommeded for use.** Normalization/Standartization required for SVM, LogReg etc.
- **df_modelling_dimensionless.xlsx** - df_modelling_no_multicollinearity.xlsx with dropped 'liquid_density', 'surface_tension', 'viscosity', 'droplet_diameter' as dimension-features. This dataframe can be used to test the possibility to create more generalized model.

## Source dataframes:
- *archive/suspension_experiments.xlsx* - cut version of the original experimental dataset, with suspensions and substrates. Some unnecessary data were removed:
    - "Время установки картриджа"
    - "Время генерации капли"
    - "Количество импульсов": min 45, max 50
    - "Длительность миниимпульсов, мс": 5
    - "Расстояние между импульсами, мс": 5
    - "Давление, мбар": 1500
    - "Насадка": 'зел'
    - "Наличие видео"
    - "Дата эксперимента"
    - "Комментарии"

First two columns allow to calculate average time without mixing before the droplet generating. Min time: 27 s, Median: 66 s, Average: 84 s
- *archive/Substrates.xlsx* - update of the substrates sheet in suspension_experiments.xlsx. Used to fill NaNs in roughness.

## Unused dataframes and files:
- *archive/df_merged* - old version of the main dataset (with errors)
- *archive/df_merged_edited.xlsx* - do not used since newest version df_main.xlsx is used (with errors)
- *archive/data_target.xlsx* - was merged to the df_merged.xlsx (with errors)
- *archive/df_na.xlsx*, *old/df_na_first.xlsx*, *old/df_na_second.xlsx* - service datasets, used to fill NaNs in droplet diameters
- *archive/report.html* - old version of the report_edited.html
- *archive/report_edited.html* - pandas_profiling report of the *df_merged_edited.xlsx*, which was be basically the same as the *df_main.xlsx*, before editing labels (with errors).
- *archive/Suspension Data Labeling.xlsx* - old version of the Suspension Data Labeling Edited.xlsx (both with errors)
- *archive/_check_breaking_up_and_rebound.xlsx* - was created in *../notebooks/edit_labels.ipynb* to check correctness.
