from CheetahPyAnalytics import (
    fetch_new_dataset,
    dataset_preprocess,
    metric_functions
)
import numpy as np
import pandas as pd


def test_metric_functions_activity():
    frame = pd.DataFrame({'power':[0,1,2,3,4,5]})
    zone_cutoffs = [0,2,4]
    mets = metric_functions()
    values = mets.activity_metric(
        frame=frame, 
        metric_name='TIZ', 
        zone_cutoffs=zone_cutoffs)
    assert values.sum() == 6

def test_metric_functions_activity_summary():
    frame = pd.DataFrame({'NP':[250]})
    FTP = 300
    mets = metric_functions()
    value = mets.activity_summary_metric(
        frame=frame,
        metric_name='IF',
        FTP=FTP)
    assert value.values == [250.0/300.0]


# if __name__ == '__main__':
#     metric_functions_activity_test()
#     metric_functions_activity_summary_test()

#     print("Everything passed")