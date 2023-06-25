from fitlitics import (
    fetch_new_dataset,
    dataset_preprocess,
    metric_functions,
    athlete,
    athlete_loadperf,
    athlete_pmc,
    model_classic_pmc
)

import numpy as np
import pandas as pd

def test_loadperf_dataclass():
    load_series = pd.Series([60,80,60])
    alp = athlete_loadperf(load_data=load_series)
    assert alp.load_data[1] == 80

def test_athlete_pmc():
    ctl = pd.Series([100,100,100])
    atl = pd.Series([50,100,150])
    pmc = athlete_pmc(ctl = ctl, atl = atl)
    assert pmc.tsb[1] == 0

def test_model_classic_pmc():
    load_series = pd.Series([100,100,100,60,80,60,60,80,60,60,80,60])
    pmc = model_classic_pmc(load_data=load_series, ctl_days=42, atl_days=7)
    pmc.run()
    # accute should be lower than chronic
    assert pmc.ath_pmc.tsb[10] > 0
