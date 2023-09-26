# Data Folder

## Essential dataframes and files:
- **df_main.xlsx** - main experimental dataset used for the final modelling (branch "modelling")
- **Suspension Data Labeling Edited.xlsx** - dataset with correct preliminary labels. 
This dataset was used to create a final labels, such as 'splashing', 'breaking_up', 'net_impact', 'rebound'.
See *../notebooks/EDA update.ipynb*.
Now it can be used to expand the research results
- **report_edited.html** - pandas_profiling report of the *df_merged_edited.xlsx*, which must be basically the same as the *df_main.xlsx*.

## Source dataframes:
- *suspension_experiments.xlsx* - cut version of the original experimental dataset, with suspensions and substrates. Some unnecessary data were removed:
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
- *Substrates.xlsx* - update of the substrates sheet in suspension_experiments.xlsx. Used to fill NaNs in roughness.

## Unused dataframes and files:
- *df_merged_edited.xlsx* - since newest version df_main.xlsx is used
- *df_merged* - old version of the main dataset (with errors)
- *data_target.xlsx* - was merged to the df_merged.xlsx
- *df_na.xlsx*, *df_na_first.xlsx*, *df_na_second.xlsx* - service datasets, used to fill NaNs in droplet diameters
- *report.html* - old version of the report_edited.html
- *Suspension Data Labeling.xlsx* - old version of the Suspension Data Labeling Edited.xlsx

