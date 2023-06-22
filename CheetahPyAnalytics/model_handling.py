import numpy as np
import pandas as pd
from CheetahPyAnalytics import metric_functions

from dataclasses import dataclass, field

@dataclass
class athlete_load:
    load_data: pd.DataFrame
    load_metric: str


@dataclass
class athelte_pmc:
    ctl: pd.Series
    atl: pd.Series
    tsb: float = field(init=False)

    def __post_init__(self):
        self.tsb = self.ctl - self.atl


class model_classic_pmc:
    def __init__(self, load_data:pd.DataFrame, load_metric:str, ctl_days:int, atl_days:int):
        self.athlete_load = athlete_load(load_data=load_data, load_metric=load_metric)
        self.ctl_days = ctl_days
        self.atl_days = atl_days
    
    def change_load_metric(self, load_metric:str):
        assert load_metric in self.athlete_load.load_data.columns, "`load_metric` must be a current column in data"
        self.athlete_load.load_metric = load_metric
        return f'Load metric set to {load_metric}'
    
    def calculate_ema(self, raw_load:pd.Series, k_days:float):
        # current = (1-k) * Yesterday’s CTL + k * Today’s TSS
        ema = raw_load.ewm(com=k_days)
        return ema

    def calculate_training_load_vectors(self):
        raw_load = self.athlete_load.load_data[self.athlete_load.load_metric]
        ctl = self.calculate_ema(raw_load=raw_load, k_days=self.ctl_days)
        atl = self.calculate_ema(raw_load=raw_load, k_days=self.atl_days)
        ath_pmc = athelte_pmc(ctl=ctl, atl=atl)
        return ath_pmc
