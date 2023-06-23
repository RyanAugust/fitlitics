# import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class athlete_loadperf:
    """Holds data classified as load and performance for transfer to models
    load is required for all models while preformace data is only used with
    models that utilize a fit."""
    load_data: pd.Series
    performance_data: pd.Series = None


@dataclass
class athlete_pmc:
    """data class for storage of cronic, accute, and balance of load metric"""
    ctl: pd.Series
    atl: pd.Series
    tsb: float = field(init=False)

    def __post_init__(self):
        self.tsb = self.ctl - self.atl


class model_classic_pmc:
    """Implementation of classic Performance Management Chart logic.
    Takes the input of load_data and cronic/accute lookback day settings
    to calculate the running CTL and ATL metrics for an athlete
    
    Standard lookback settings are CTL:42 and ATL:7 
    However, these should be further customized to each athlete as soon as performance
    data is available"""
    def __init__(self, load_data: pd.Series, ctl_days: int = 42, atl_days: int = 7):
        self.athlete_loadperf = athlete_loadperf(load_data=load_data)
        self.ctl_days = ctl_days
        self.atl_days = atl_days
    
    def _calculate_ema(self, raw_load: pd.Series, k_days: float):
        # current = (1-k) * Yesterday’s CTL + k * Today’s TSS
        ema = raw_load.ewm(com=k_days)
        return ema

    def _calculate_training_load_vectors(self):
        ctl = self._calculate_ema(raw_load=self.athlete_loadperf.load_data, k_days=self.ctl_days)
        atl = self._calculate_ema(raw_load=self.athlete_loadperf.load_data, k_days=self.atl_days)
        ath_pmc = athlete_pmc(ctl=ctl, atl=atl)
        return ath_pmc
    
    def fit(self):
        assert "This model does not require fitting of performance data to load data"

    def run(self):
        ath_pmc = self._calculate_training_load_vectors()
        return ath_pmc