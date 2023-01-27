import numpy as np
import pandas as pd
import math
from numpy.random import RandomState
from scipy.stats import norm

from sklearn.metrics.cluster import adjusted_rand_score

def consensus_score(list_arr_labels):
    """
    Parameters:
    `list_arr_labels`: list of labels arrays
    """

    scr = []
    for i in range(len(list_arr_labels)):
        for j in range(i+1, len(list_arr_labels)):
            scr.append(adjusted_rand_score(list_arr_labels[i], list_arr_labels[j]))
    return sum(scr)
    
    
def get_rate(N, k, n_min):
    """
    Parameters:
    `N` int > 1: approximate number of examples
    `k` int > 1: number of clusters
    `n_min` int: minimum number of examples per cluster
    """
    
    assert type(N) == int and N > 1
    assert type(k) == int and k > 1
    
    rate_c = []
    resto = N
    for j in range(2, k+2):
        rate_c.append(int(resto/j))
        resto = N - sum(rate_c)
    rate_s = [int(sum(rate_c)/k) for i in range(k)]
    rate=[rate_s, rate_c]
    
    assert (min(rate_s) and min(rate_c)) >= n_min
    
    return(rate)
