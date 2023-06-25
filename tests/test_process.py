from fitlitics import (
    fetch_new_dataset,
    dataset_preprocess,
    metric_functions,
    athlete
)

import numpy as np
import pandas as pd

athlete_statics = athlete(
    weight = 85,
    max_heart_rate = 191,
    resting_heart_rate = 40,
    ae_threshold_heart_rate = 148,
    lactate_threshold_heart_rate = 168,
    run_critical_power = 356,
    run_w_prime = 16900,
    run_p_max = 642,
    bike_functional_threshold_power = 300)


def test_preprocess():
    frame = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', end='2023-01-03'),
        'normalized_power':[200,200,200],
        'intensity_factor':[.90,.90,.90],
        'duration':[100,120,150],
        'functional_threshold_power':[223,223,223],
        'heart_rate':[110,110,109],
        'average_heart_rate':[110,110,109],
        'power':[200,200,200],
        'average_power':[200,200,200],
        'sport':['Bike','Bike','Bike'],
        'Sport':['Bike','Bike','Bike'],
        'pace':[np.nan,np.nan,np.nan],
        'weight':[85,85,85]})
    frame.to_csv('./local_activity_store.csv', index=False)
    load_metric = 'TSS'
    performance_metric = 'VO2'
    dp = dataset_preprocess(
                       local_activity_store='./local_activity_store.csv',
                       athlete_statics=athlete_statics)
    dp.pre_process(load_metric=load_metric,
                   performance_metric=performance_metric,
                   performance_lower_bound = 0.0,
                   sport = False,
                   fill_performance_forward=True)
