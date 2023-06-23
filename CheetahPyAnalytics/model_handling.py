# import numpy as np
import pandas as pd

from dataclasses import dataclass, field

@dataclass
class athlete_loadperf:
    load_data: pd.Series
    performance_data: pd.Series = None


@dataclass
class athlete_pmc:
    ctl: pd.Series
    atl: pd.Series
    tsb: float = field(init=False)

    def __post_init__(self):
        self.tsb = self.ctl - self.atl


class model_classic_pmc:
    def __init__(self, load_data: pd.DataFrame, load_metric: str, ctl_days: int, atl_days: int):
        self.athlete_loadperf = athlete_loadperf(load_data=load_data)
        self.ctl_days = ctl_days
        self.atl_days = atl_days
    
    def change_load_metric(self, load_metric: str):
        assert load_metric in self.athlete_loadperf.load_data.columns, "`load_metric` must be a current column in data"
        self.athlete_loadperf.load_metric = load_metric
        return f'Load metric set to {load_metric}'
    
    def calculate_ema(self, raw_load: pd.Series, k_days: float):
        # current = (1-k) * Yesterday’s CTL + k * Today’s TSS
        ema = raw_load.ewm(com=k_days)
        return ema

    def calculate_training_load_vectors(self):
        ctl = self.calculate_ema(raw_load=self.athlete_loadperf.load_data, k_days=self.ctl_days)
        atl = self.calculate_ema(raw_load=self.athlete_loadperf.load_data, k_days=self.atl_days)
        ath_pmc = athlete_pmc(ctl=ctl, atl=atl)
        return ath_pmc
