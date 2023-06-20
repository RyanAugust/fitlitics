__title__ = 'CheetahPyAnalytics'
__version__ = '0.2.3'
__author__ = 'RyanAugust'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023'

from .cheetahpyanalytics import (
    fetch_new_dataset,
    dataset_preprocess
    )
from .functions import metric_functions
from .model_handling import (
    athlete_load,
    athelte_pmc,
    model_classic_pmc
)
__all__ = [
    'fetch_new_dataset',
    'dataset_preprocess',
    'metric_functions',
    'athlete_load',
    'athelte_pmc',
    'model_classic_pmc'
]

metric_funcs = metric_functions()