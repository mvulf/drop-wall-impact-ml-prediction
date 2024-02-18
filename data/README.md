# Data Folder

## Essential dataframes and files:
- **df_main.xlsx** - main experimental dataset
Old dataset *"old_df_main.xlsx"* - old version of the main experimental dataset, which were used for the final modelling (branch "modelling").
- **Suspension Data Labeling_We_Re_edit.xlsx** - dataset with correct preliminary labels. 
This dataset was used to create a final labels, such as 'splashing', 'breaking_up', 'net_impact', 'rebound'.
See *../notebooks/edit_labels.ipynb*.
Old version is: *"old_Suspension Data Labeling Edited.xlsx"* (see *../notebooks/EDA update.ipynb*).

FOR MODELLING:
- **df_modelling_full.xlsx** - dataframe whith dropped droplet generation parameters ('one_drop', 'voltage', 'long_impulse_duration', 'long_impulse_dur_binary'), test number ('test'), 'splashing_spectrum', 'breaking_up', 'rebound', 'height', 'particle_diameter_cat', 'volume_fraction'. However, there are still multicorrelated parameters. **It is not recommended for use**.
- **df_modelling_no_multicollinearity.xlsx** - df_modelling_full.xlsx with dropped 'particle_mean_diameter', 'particle_density', 'roughness', 'velocity' as highly correlated with 'particle_droplet_diameter_ratio', 'particle_liquid_density_ratio', 'roughness_binary', 'We' respectively. **Highly recommeded for use.** Normalization/Standartization required for SVM, LogReg etc.
- **df_modelling_dimensionless.xlsx** - df_modelling_no_multicollinearity.xlsx with dropped 'liquid_density', 'surface_tension', 'viscosity', 'droplet_diameter' as dimension-features. This dataframe can be used to test the possibility to create more generalized model.


## Source dataframes:
- *old/suspension_experiments.xlsx* - cut version of the original experimental dataset, with suspensions and substrates. Some unnecessary data were removed:
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
- *old/Substrates.xlsx* - update of the substrates sheet in suspension_experiments.xlsx. Used to fill NaNs in roughness.

## Unused dataframes and files:
- *old/df_merged_edited.xlsx* - do not used since newest version df_main.xlsx is used
- *old/df_merged* - old version of the main dataset (with errors)
- *old/data_target.xlsx* - was merged to the df_merged.xlsx
- *old/df_na.xlsx*, *old/df_na_first.xlsx*, *old/df_na_second.xlsx* - service datasets, used to fill NaNs in droplet diameters
- *old/report.html* - old version of the report_edited.html
- *old/report_edited.html* - pandas_profiling report of the *df_merged_edited.xlsx*, which was be basically the same as the *df_main.xlsx*, before editing labels.
- *old/Suspension Data Labeling.xlsx* - old version of the Suspension Data Labeling Edited.xlsx
- *_check_breaking_up_and_rebound.xlsx* - was created in *../notebooks/edit_labels.ipynb* to check correctness.
