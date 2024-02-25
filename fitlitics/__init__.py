__title__ = 'fitlitics'
__author__ = 'RyanAugust'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023'

import os

with open(os.path.join(os.path.dirname(__file__), "VERSION")) as version_file:
    __version__ = version_file.read().strip()

from .CheetahPyAnalytics import (
    fetch_new_dataset,
    dataset_preprocess
    )
from .functions import (
    metric_functions,
    athlete
)
from .model_handling import (
    athlete_loadperf,
    athlete_pmc,
    model_classic_pmc
)
__all__ = [
    'fetch_new_dataset',
    'dataset_preprocess',
    'athlete',
    'metric_functions',
    'athlete_loadperf',
    'athlete_pmc',
    'model_classic_pmc'
]

metric_funcs = metric_functions()
