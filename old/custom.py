# testing.py
# some functions for partitioning a dataset
#
# Author: Ronny Macmaster

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.base import BaseEstimator, TransformerMixin

def part(data, ratio):
    """Partition the test set. (train, test)"""
    shuffle = np.random.permutation(len(data))
    pivot = int(len(data) * ratio)
    train = data.iloc[shuffle[pivot:]]
    test = data.iloc[shuffle[:pivot]]
    return train, test 

def strat_part(data, ratio, column):
    """Stratify partition the test set based on a column. (train, test)."""
    split = StratifiedShuffleSplit(n_splits=1, test_size=ratio, random_state=42)
    for train_idx, test_idx in split.split(data, data[column]):
       train = data.loc[train_idx] 
       test = data.loc[test_idx]
    return train, test


