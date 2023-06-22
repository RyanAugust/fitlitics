from CheetahPyAnalytics import (
    fetch_new_dataset,
    dataset_preprocess,
    metric_functions,
    athlete_statics
)
import numpy as np
import pandas as pd


def test_metric_functions_activity():
    frame = pd.DataFrame({'power':[0,1,2,3,4,5]})
    zone_cutoffs = [0,2,4]
    metric_funcs = metric_functions()
    values = metric_funcs.activity_metric(
        frame=frame,
        metric_name='TIZ',
        zone_cutoffs=zone_cutoffs)
    assert values.sum() == 6

def test_metric_functions_activity_summary():
    frame = pd.DataFrame({'normalized_power':[250]})
    ath_statics = athlete_statics(functional_threshold_power = 300)
    metric_funcs = metric_functions()
    value = metric_funcs.activity_summary_metric(
        frame=frame,
        metric_name='IF',
        athlete_statics=athlete_statics)
    assert value.values == [250.0/300.0]

def test_activity_VO2_calc():
    frame = pd.DataFrame({'power':[200,200,200,200],
                          'heart_rate':[110,110,110,110],
                          'pace':[14,14,14,14]})
    sport = 'Bike'
    ath_statics = athlete_statics(
        resting_heart_rate = 40,
        max_heart_rate = 190,
        athlete_mass = 85)
    metric_funcs = metric_functions()
    value = metric_funcs.activity_metric(
        frame=frame,
        metric_name='VO2',
        sport = sport)
    assert value == [((200/75*1000)/85)/((110-40)/(190-40))]


# if __name__ == '__main__':
#     metric_functions_activity_test()
#     metric_functions_activity_summary_test()

#     print("Everything passed")