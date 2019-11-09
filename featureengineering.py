# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from load_all_datasets import compile_datasets

def correct_shift(datasets):
    a = datasets["Last On Drug Day"] 
    b = datasets["On Drug Start Day"]
    var = a-b if a != 0 else b
    return var

def add_features(list_of_dfs):
    for index, df in enumerate(list_of_dfs):
        df["Enrollment Time"] = df["Day Enrollment Completed"] - df["Day Enrollment Received"]
        df["On Drug Time"] = df[["Last On Drug Day", "On Drug Start Day"]].apply(correct_shift, axis=1)
        df["# of Payment Methods"] = df[["Payment Method #4", "Payment Method #5", "Payment Method #1", "Payment Method #3", "Payment Method #2"]].sum(axis=1)
        list_of_dfs[index] = df
        
    return list_of_dfs


