__title__ = 'CheetahPyAnalytics'
__version__ = '0.2.3'
__author__ = 'RyanAugust'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023'

import os
import datetime
import requests
import math
import json

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


from .CheetahPyAnalytics import (
    fetch_new_dataset
    ,dataset_preprocess
    ,load_functions
    ,performance_functions)


__all__ = [
    'dataset',
    'dataset_preprocess',
    'load_functions',
    'performance_functions'
]
